use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, RwLock};
use std::task::{Context, Poll};

use event_listener::{Event, EventListener};
use pin_project_lite::pin_project;

use crate::clock::ClockDomain;
use crate::executor;

/// Explicit timeout policy required by async waits so infinite waits must be opt-in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WaitTimeout {
    /// Wait using `ClockDomain::World` milliseconds (pauses when world clock is 0.0).
    WorldMs(u64),
    /// Wait using `ClockDomain::Ui` milliseconds (always advances).
    UiMs(u64),
    /// Explicitly opt into waiting without a timeout.
    None,
}

/// Outcome failure for bounded async channel or signal waits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WaitError {
    /// The specified `WaitTimeout` elapsed before an item or state matched.
    Timeout,
    /// The underlying channel was closed.
    Closed,
    /// A timed wait was polled outside an active `TaskSlab` simulation tick.
    NoClockContext,
}

impl std::fmt::Display for WaitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Timeout => write!(
                f,
                "wait timed out"
            ),
            Self::Closed => write!(
                f,
                "channel closed"
            ),
            Self::NoClockContext => write!(
                f,
                "timed wait polled without clock context"
            ),
        }
    }
}

impl std::error::Error for WaitError {}

/// FIFO sender handle for discrete gameplay and UI actions.
#[derive(Debug)]
pub struct ActionSender<T> {
    inner: async_channel::Sender<T>,
}

impl<T> Clone for ActionSender<T> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

/// FIFO receiver handle for discrete gameplay and UI actions.
#[derive(Debug)]
pub struct ActionReceiver<T> {
    inner: async_channel::Receiver<T>,
}

impl<T> Clone for ActionReceiver<T> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

/// Creates a bounded FIFO action channel.
pub(crate) fn action_channel<T>(
    capacity: usize,
) -> (
    ActionSender<T>,
    ActionReceiver<T>,
) {
    let (tx, rx) = async_channel::bounded(capacity.max(1));
    (
        ActionSender { inner: tx },
        ActionReceiver { inner: rx },
    )
}

impl<T> ActionSender<T> {
    /// Enqueues an action without blocking; drops if channel is full and returns `Err`.
    pub fn try_send(&self, action: T) -> Result<(), T> {
        self.inner.try_send(action).map_err(
            |err| match err {
                async_channel::TrySendError::Full(val)
                | async_channel::TrySendError::Closed(val) => val,
            },
        )
    }

    /// Asynchronously sends an action into the FIFO queue.
    pub async fn send(&self, action: T) -> Result<(), WaitError> {
        self.inner.send(action).await.map_err(|_| WaitError::Closed)
    }

    /// Returns the number of queued actions waiting in the channel.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Returns `true` if no actions are currently queued.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

impl<T> ActionReceiver<T> {
    /// Pops an immediately available action without awaiting.
    pub fn try_recv(&self) -> Option<T> {
        self.inner.try_recv().ok()
    }

    /// Awaits the next FIFO action, bounded by an explicit `WaitTimeout`.
    pub fn next(&self, timeout: WaitTimeout) -> impl Future<Output = Result<T, WaitError>> + '_ {
        ActionNextFuture {
            recv: Box::pin(self.inner.recv()),
            timer: SleepFuture::from_timeout(timeout),
        }
    }
}

pin_project! {
    struct ActionNextFuture<F> {
        #[pin]
        recv: F,
        #[pin]
        timer: Option<SleepFuture>,
    }
}

impl<F, T> Future for ActionNextFuture<F>
where
    F: Future<Output = Result<T, async_channel::RecvError>>,
{
    type Output = Result<T, WaitError>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let mut this = self.project();
        if let Poll::Ready(res) = this.recv.as_mut().poll(cx) {
            return Poll::Ready(res.map_err(|_| WaitError::Closed));
        }
        if let Some(timer) = this.timer.as_mut().as_pin_mut() {
            match timer.poll(cx) {
                Poll::Ready(Ok(())) => return Poll::Ready(Err(WaitError::Timeout)),
                Poll::Ready(Err(e)) => return Poll::Ready(Err(e)),
                Poll::Pending => {}
            }
        }
        Poll::Pending
    }
}

struct SignalInner<T> {
    value: RwLock<T>,
    event: Event,
}

/// Shared state container that prevents holding read locks across `.await`.
pub struct StateSignal<T> {
    inner: Arc<SignalInner<T>>,
}

impl<T> Clone for StateSignal<T> {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl<T: std::fmt::Debug> std::fmt::Debug for StateSignal<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StateSignal").finish_non_exhaustive()
    }
}

impl<T> StateSignal<T> {
    /// Creates a new `StateSignal` initialized with `initial`.
    pub fn new(initial: T) -> Self {
        Self {
            inner: Arc::new(
                SignalInner {
                    value: RwLock::new(initial),
                    event: Event::new(),
                },
            ),
        }
    }

    /// Reads the state synchronously inside a closure so no lock guard can cross `.await`.
    pub fn read<R>(&self, f: impl FnOnce(&T) -> R) -> R {
        let guard = self
            .inner
            .value
            .read()
            .unwrap_or_else(|e| panic!("StateSignal read lock poisoned: {e}"));
        f(&*guard)
    }

    /// Mutates the state synchronously and notifies all pending `wait_until` futures.
    pub fn update(&self, f: impl FnOnce(&mut T)) {
        {
            let mut guard = self
                .inner
                .value
                .write()
                .unwrap_or_else(|e| panic!("StateSignal write lock poisoned: {e}"));
            f(&mut *guard);
        }
        self.inner.event.notify(usize::MAX);
    }

    /// Replaces the state value and notifies all pending `wait_until` futures.
    pub fn set(&self, value: T) {
        self.update(|slot| *slot = value);
    }

    /// Waits until `predicate` returns `Some(R)` or the `WaitTimeout` expires.
    pub fn wait_until<'a, R>(
        &'a self,
        timeout: WaitTimeout,
        predicate: impl FnMut(&T) -> Option<R> + 'a,
    ) -> impl Future<Output = Result<R, WaitError>> + 'a {
        WaitUntilFuture {
            signal: self,
            predicate,
            listener: None,
            timer: SleepFuture::from_timeout(timeout),
        }
    }
}

pin_project! {
    struct WaitUntilFuture<'a, T, P> {
        signal: &'a StateSignal<T>,
        predicate: P,
        listener: Option<EventListener>,
        #[pin]
        timer: Option<SleepFuture>,
    }
}

impl<T, P, R> Future for WaitUntilFuture<'_, T, P>
where
    P: FnMut(&T) -> Option<R>,
{
    type Output = Result<R, WaitError>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let mut this = self.project();

        loop {
            if let Some(result) = this.signal.read(|val| (this.predicate)(val)) {
                return Poll::Ready(Ok(result));
            }

            if let Some(timer) = this.timer.as_mut().as_pin_mut() {
                match timer.poll(cx) {
                    Poll::Ready(Ok(())) => return Poll::Ready(Err(WaitError::Timeout)),
                    Poll::Ready(Err(e)) => return Poll::Ready(Err(e)),
                    Poll::Pending => {}
                }
            }

            if this.listener.is_none() {
                // Register listener before re-checking predicate to prevent missed notifications.
                *this.listener = Some(this.signal.inner.event.listen());
                continue;
            }

            let listener = this.listener.as_mut().expect("listener checked above");
            match Pin::new(listener).poll(cx) {
                Poll::Ready(()) => {
                    *this.listener = None;
                }
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

/// Future that completes when the target clock domain reaches the computed deadline.
pub(crate) struct SleepFuture {
    domain: ClockDomain,
    duration_ms: u64,
    deadline_ms: Option<u64>,
}

impl SleepFuture {
    /// Creates a `SleepFuture` in the given clock domain for `duration_ms`.
    pub(crate) fn new(domain: ClockDomain, duration_ms: u64) -> Self {
        Self {
            domain,
            duration_ms,
            deadline_ms: None,
        }
    }

    /// Converts a `WaitTimeout` into an optional `SleepFuture`.
    pub(crate) fn from_timeout(timeout: WaitTimeout) -> Option<Self> {
        match timeout {
            WaitTimeout::WorldMs(ms) => Some(
                Self::new(
                    ClockDomain::World,
                    ms,
                ),
            ),
            WaitTimeout::UiMs(ms) => Some(
                Self::new(
                    ClockDomain::Ui,
                    ms,
                ),
            ),
            WaitTimeout::None => None,
        }
    }
}

impl Future for SleepFuture {
    type Output = Result<(), WaitError>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let Some(now_ms) = executor::current_domain_time_ms(self.domain) else {
            return Poll::Ready(Err(WaitError::NoClockContext));
        };

        let deadline = match self.deadline_ms {
            Some(d) => d,
            None => {
                let d = now_ms.saturating_add(self.duration_ms);
                self.deadline_ms = Some(d);
                d
            }
        };

        if now_ms >= deadline {
            Poll::Ready(Ok(()))
        } else {
            executor::register_current_timer(
                self.domain,
                deadline,
                cx.waker().clone(),
            );
            Poll::Pending
        }
    }
}
