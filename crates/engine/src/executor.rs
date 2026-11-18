use std::any::Any;
use std::cell::RefCell;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::task::{Context, Poll, Wake, Waker};

use async_executor::Executor;

use crate::clock::{ClockDomain, DualClocks};

/// Trait object bridge allowing `TaskSlab` timers and `ScriptContext` to access
/// the active simulation step synchronously without exposing storage internals or mutexes.
pub(crate) trait SimAccess: Any {
    fn domain_time_ms(&self, domain: ClockDomain) -> u64;
    fn register_timer(&mut self, domain: ClockDomain, deadline_ms: u64, waker: Waker);
    fn as_any_mut(&mut self) -> &mut dyn Any;
    fn into_any(self: Box<Self>) -> Box<dyn Any>;
}

thread_local! {
    static ACTIVE_SIM: RefCell<Option<Box<dyn SimAccess>>> = const { RefCell::new(None) };
}

struct ActiveSimGuard;

impl Drop for ActiveSimGuard {
    fn drop(&mut self) {
        // Ensure thread-local state does not leak if a panic occurs mid-poll.
        ACTIVE_SIM.with(
            |cell| {
                let _ = cell.borrow_mut().take();
            },
        );
    }
}

/// Binds `sim` for the duration of a synchronous `TaskSlab` poll pass using safe Rust.
pub(crate) fn with_active_sim<S: SimAccess + 'static, R>(
    sim: Box<S>,
    f: impl FnOnce() -> R,
) -> (
    Box<S>,
    R,
) {
    ACTIVE_SIM.with(
        |cell| {
            let mut slot = cell.borrow_mut();
            *slot = Some(sim);
        },
    );
    let guard = ActiveSimGuard;
    let out = f();
    std::mem::forget(guard);
    let restored_dyn = ACTIVE_SIM
        .with(|cell| cell.borrow_mut().take())
        .expect("active sim missing after poll pass");
    let restored = restored_dyn.into_any().downcast::<S>().unwrap_or_else(
        |e| {
            panic!(
                "Active sim type mismatch for {}: dynamic type id {:?}",
                std::any::type_name::<S>(),
                (*e).type_id()
            )
        },
    );
    (
        restored, out,
    )
}

pub(crate) fn with_sim_mut<T: 'static, R>(f: impl FnOnce(&mut T) -> R) -> Option<R> {
    ACTIVE_SIM.with(
        |cell| {
            let mut guard = cell.borrow_mut();
            let sim_ref = guard.as_deref_mut()?;
            let typed = sim_ref.as_any_mut().downcast_mut::<T>()?;
            Some(f(typed))
        },
    )
}

pub(crate) fn current_domain_time_ms(domain: ClockDomain) -> Option<u64> {
    ACTIVE_SIM.with(
        |cell| {
            let guard = cell.borrow();
            let sim_ref = guard.as_deref()?;
            Some(sim_ref.domain_time_ms(domain))
        },
    )
}

pub(crate) fn register_current_timer(domain: ClockDomain, deadline_ms: u64, waker: Waker) {
    ACTIVE_SIM.with(
        |cell| {
            let mut guard = cell.borrow_mut();
            if let Some(sim_ref) = guard.as_deref_mut() {
                sim_ref.register_timer(
                    domain,
                    deadline_ms,
                    waker,
                );
            }
        },
    );
}

#[derive(Clone)]
struct SlotWakeFlag {
    woken: Arc<AtomicBool>,
}

impl Wake for SlotWakeFlag {
    fn wake(self: Arc<Self>) {
        self.woken.store(
            true,
            Ordering::Release,
        );
    }

    fn wake_by_ref(self: &Arc<Self>) {
        self.woken.store(
            true,
            Ordering::Release,
        );
    }
}

/// Owner-bound handle stored privately by `engine` (`ActorRecord`, `ModeLayer`, `WorldState`).
/// Dropping `TaskHandle` cancels the script on the next `TaskSlab` poll pass.
#[derive(Debug)]
pub(crate) struct TaskHandle {
    cancelled: Arc<AtomicBool>,
}

impl TaskHandle {
    pub(crate) fn new(cancelled: Arc<AtomicBool>) -> Self {
        Self { cancelled }
    }
}

impl Drop for TaskHandle {
    fn drop(&mut self) {
        self.cancelled.store(
            true,
            Ordering::Release,
        );
    }
}

struct TaskSlot {
    future: Pin<Box<dyn Future<Output = ()> + Send>>,
    woken: Arc<AtomicBool>,
    waker: Waker,
    cancelled: Arc<AtomicBool>,
}

/// Single-threaded gameplay script executor with deterministic sorted-index polling
/// and zero gameplay mutexes.
pub(crate) struct TaskSlab {
    slots: Vec<Option<TaskSlot>>,
    free_indices: Vec<usize>,
    active_indices: Vec<usize>,
}

impl TaskSlab {
    pub(crate) fn new() -> Self {
        Self {
            slots: Vec::new(),
            free_indices: Vec::new(),
            active_indices: Vec::new(),
        }
    }

    /// Spawns a future with a pre-created cancellation token owned by `ActorRecord` or `ModeLayer`.
    pub(crate) fn spawn_with_token(
        &mut self,
        future: Pin<Box<dyn Future<Output = ()> + Send>>,
        cancelled: Arc<AtomicBool>,
    ) -> TaskHandle {
        let woken = Arc::new(AtomicBool::new(true));
        let waker = Waker::from(
            Arc::new(
                SlotWakeFlag {
                    woken: Arc::clone(&woken),
                },
            ),
        );

        let slot = TaskSlot {
            future,
            woken,
            waker,
            cancelled: Arc::clone(&cancelled),
        };

        let index = if let Some(idx) = self.free_indices.pop() {
            self.slots[idx] = Some(slot);
            idx
        } else {
            let idx = self.slots.len();
            self.slots.push(Some(slot));
            idx
        };

        // Keep active indices sorted so polling order is deterministic.
        let insert_pos = self
            .active_indices
            .binary_search(&index)
            .unwrap_or_else(|p| p);
        self.active_indices.insert(
            insert_pos, index,
        );

        TaskHandle { cancelled }
    }

    /// Polls all ready tasks in ascending sorted index order until quiescence (bounded).
    pub(crate) fn poll_ready<S: SimAccess + 'static>(&mut self, sim: Box<S>) -> Box<S> {
        let (restored, ()) = with_active_sim(
            sim,
            || {
                // Multiple passes allow tasks woken synchronously within the same step to advance.
                for _pass in 0..8 {
                    let mut any_progressed = false;
                    let mut i = 0;
                    while i < self.active_indices.len() {
                        let idx = self.active_indices[i];
                        let Some(slot) = self.slots[idx].as_mut() else {
                            self.active_indices.remove(i);
                            continue;
                        };

                        if slot.cancelled.load(Ordering::Acquire) {
                            self.slots[idx] = None;
                            self.free_indices.push(idx);
                            self.active_indices.remove(i);
                            continue;
                        }

                        if !slot.woken.swap(
                            false,
                            Ordering::AcqRel,
                        ) {
                            i += 1;
                            continue;
                        }

                        any_progressed = true;
                        let waker = slot.waker.clone();
                        let mut cx = Context::from_waker(&waker);
                        match slot.future.as_mut().poll(&mut cx) {
                            Poll::Ready(()) => {
                                self.slots[idx] = None;
                                self.free_indices.push(idx);
                                self.active_indices.remove(i);
                            }
                            Poll::Pending => {
                                i += 1;
                            }
                        }
                    }

                    if !any_progressed {
                        break;
                    }
                }
            },
        );
        restored
    }
}

/// Future handle for background compute work executed on the `async-executor` worker pool.
pub struct ComputeTask<T> {
    inner: async_executor::Task<T>,
}

impl<T> Future for ComputeTask<T> {
    type Output = T;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        Pin::new(&mut self.inner).poll(cx)
    }
}

/// Private multi-threaded `async-executor` pool for heavy compute, pathfinding, and culling.
pub(crate) struct WorkerPool {
    executor: Arc<Executor<'static>>,
    shutdown: Arc<AtomicBool>,
    workers: Vec<std::thread::JoinHandle<()>>,
}

impl WorkerPool {
    pub(crate) fn new(thread_count: usize) -> Self {
        let executor = Arc::new(Executor::new());
        let shutdown = Arc::new(AtomicBool::new(false));
        let count = thread_count.max(1);
        let mut workers = Vec::with_capacity(count);

        for idx in 0..count {
            let exec = Arc::clone(&executor);
            let stop = Arc::clone(&shutdown);
            let handle = std::thread::Builder::new()
                .name(format!("engine-worker-{idx}"))
                .spawn(
                    move || {
                        while !stop.load(Ordering::Acquire) {
                            if !exec.try_tick() {
                                std::thread::park_timeout(std::time::Duration::from_millis(2));
                            }
                        }
                    },
                )
                .unwrap_or_else(|e| panic!("Failed to spawn engine worker thread: {e}"));
            workers.push(handle);
        }

        Self {
            executor,
            shutdown,
            workers,
        }
    }

    pub(crate) fn spawn<T: Send + 'static>(
        &self,
        future: impl Future<Output = T> + Send + 'static,
    ) -> ComputeTask<T> {
        let task = self.executor.spawn(future);
        for handle in &self.workers {
            handle.thread().unpark();
        }
        ComputeTask { inner: task }
    }
}

impl Drop for WorkerPool {
    fn drop(&mut self) {
        self.shutdown.store(
            true,
            Ordering::Release,
        );
        for handle in self.workers.drain(..) {
            handle.thread().unpark();
            if let Err(e) = handle.join() {
                tracing::error!(
                    ?e,
                    "Engine worker thread panicked during shutdown"
                );
            }
        }
    }
}

pub(crate) fn advance_clocks_and_timers(clocks: &mut DualClocks, dt_ms: u64) -> f32 {
    clocks.advance(dt_ms)
}
