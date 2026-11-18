use std::sync::{Arc, mpsc};
use std::time::Instant;

use winit::application::ApplicationHandler;
use winit::error::EventLoopError;
use winit::event::{DeviceEvent, DeviceId, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::PhysicalKey;
use winit::window::{CursorGrabMode, Window, WindowId};

use crate::effect::EffectRegistry;
use crate::executor::{TaskSlab, advance_clocks_and_timers};
use crate::input::{GameInput, InputBatch, KeyTransition, MouseTransition};
use crate::renderer::Renderer;
use crate::snapshot::{CameraSnapshot, GizmoBox, RenderSnapshot};
use crate::sync::{ActionSender, action_channel};
use crate::ui::UiOverlay;
use crate::world::{EngineContext, InputContext, SyncContext, WorldReadView, WorldState};

/// Controls whether simulation and rendering execute sequentially or overlapped across frames.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderSchedule {
    /// UI: `Low Latency`. Runs `Sim N -> Render N` sequentially with zero extra visual lag.
    Sequential,
    /// UI: `High Frame Rate`. Runs `Sim N+1` on a worker thread while main thread draws `Render N`.
    Parallel,
}

/// Window and runtime configuration for `EngineRunner`.
#[derive(Debug, Clone)]
pub struct EngineConfig {
    pub title: String,
    pub width: u32,
    pub height: u32,
    pub initial_schedule: RenderSchedule,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            title: String::from("Engine Window"),
            width: 1280,
            height: 720,
            initial_schedule: RenderSchedule::Sequential,
        }
    }
}

/// Primary game interface implemented by the `game` crate.
pub trait GameApp: Send + 'static {
    type FlatMode: Clone + PartialEq + Eq + Send + 'static;
    type StackMode: Clone + PartialEq + Eq + Send + 'static;
    type Action: Send + 'static;
    type ActorData: Clone + Send + 'static;
    type SnapshotData: Clone + Default + Send + 'static;

    /// Returns initial `(FlatMode, StackMode, world_active_predicate)`.
    #[allow(clippy::type_complexity)]
    fn initial_modes() -> (
        Self::FlatMode,
        Self::StackMode,
        fn(&Self::FlatMode, &[Self::StackMode]) -> bool,
    );

    /// Initializes world entities and initial async gameplay scripts.
    fn init(&mut self, ctx: &mut EngineContext<Self>)
    where
        Self: Sized;

    /// Returns the `ClockDomain::World` time scale when world simulation is active.
    fn world_scale(&self) -> f32;

    /// Handles discrete ordered inputs sequentially against `ModeManager`.
    fn on_input(&mut self, input: GameInput, ctx: &mut InputContext<Self>)
    where
        Self: Sized;

    /// Synchronous hook invoked outside async futures for OS/disk checkpoint save/load.
    fn on_sync(&mut self, _ctx: &mut SyncContext<Self>)
    where
        Self: Sized,
    {
    }

    /// Builds the game-specific immutable UI/render state for the completed `RenderSnapshot`.
    fn build_snapshot(
        &self,
        world: &WorldReadView<Self>,
    ) -> (
        CameraSnapshot,
        Self::SnapshotData,
    )
    where
        Self: Sized;

    /// Builds optional game-agnostic 3D gizmo primitives for the completed `RenderSnapshot`.
    fn build_gizmos(&self, _world: &WorldReadView<Self>) -> Vec<GizmoBox>
    where
        Self: Sized,
    {
        Vec::new()
    }

    /// Returns whether the engine should lock and hide the mouse cursor for the current mode.
    fn wants_cursor_grab(&self, _world: &WorldReadView<Self>) -> bool
    where
        Self: Sized,
    {
        false
    }
}

struct SimRunner<G: GameApp> {
    game: G,
    world: Option<Box<WorldState<G>>>,
    slab: TaskSlab,
    frame_index: u64,
}

impl<G: GameApp> SimRunner<G> {
    fn new(mut game: G, initial_schedule: RenderSchedule) -> Self {
        let (action_sender, action_receiver) = action_channel(512);
        let mut world = Box::new(
            WorldState::new(
                action_sender,
                action_receiver,
            ),
        );
        world.schedule = initial_schedule;
        let mut slab = TaskSlab::new();

        {
            let mut ctx = EngineContext { world: &mut world };
            game.init(&mut ctx);
        }
        Self::drain_spawns(
            &mut world, &mut slab,
        );

        Self {
            game,
            world: Some(world),
            slab,
            frame_index: 0,
        }
    }

    fn drain_spawns(world: &mut WorldState<G>, slab: &mut TaskSlab) {
        for pending in world.pending_spawns.drain(..) {
            if !pending.is_cancelled() {
                // Handle lifetime is already owned by `ActorRecord`, `ModeLayer`, or `WorldState`.
                let handle = slab.spawn_with_token(
                    pending.future,
                    pending.cancelled,
                );
                std::mem::forget(handle);
            }
        }
    }

    fn step(
        &mut self,
        real_dt_ms: u64,
        inputs: InputBatch,
        out: &mut RenderSnapshot<G::SnapshotData>,
    ) -> bool {
        let mut world = self.world.take().expect("world missing at start of step");

        // Process inputs sequentially against `ModeManager` so mode changes gate subsequent inputs.
        for input in inputs.into_ordered_inputs() {
            let mut ctx = InputContext { world: &mut world };
            self.game.on_input(
                input, &mut ctx,
            );
        }
        Self::drain_spawns(
            &mut world,
            &mut self.slab,
        );

        let speed = if world.modes.is_world_active() {
            self.game.world_scale()
        } else {
            0.0
        };
        world.clocks.set_world_speed(speed);
        let world_dt_secs = advance_clocks_and_timers(
            &mut world.clocks,
            real_dt_ms,
        );

        // Coupled simulation step: Physics -> poll TaskSlab -> flush PhysicsCommand.
        world.physics.step(world_dt_secs);
        let mut world = self.slab.poll_ready(world);
        Self::drain_spawns(
            &mut world,
            &mut self.slab,
        );
        world
            .physics
            .flush_commands(&mut world.pending_physics_cmds);

        {
            let mut sync_ctx = SyncContext { world: &mut world };
            self.game.on_sync(&mut sync_ctx);
        }
        Self::drain_spawns(
            &mut world,
            &mut self.slab,
        );

        self.frame_index = self.frame_index.wrapping_add(1);
        let (camera, game_snap, gizmos, cursor_grab) = {
            let view = world.read_view();
            let (camera, game_snap) = self.game.build_snapshot(&view);
            let gizmos = self.game.build_gizmos(&view);
            let cursor_grab = self.game.wants_cursor_grab(&view);
            (
                camera,
                game_snap,
                gizmos,
                cursor_grab,
            )
        };
        let meshes = world.collect_render_meshes(&camera);

        out.frame_index = self.frame_index;
        out.world_time_ms = world.clocks.world_time_ms();
        out.ui_time_ms = world.clocks.ui_time_ms();
        out.world_speed = world.clocks.world_speed();
        out.schedule = world.schedule;
        out.camera = camera;
        out.meshes = meshes;
        out.gizmos = gizmos;
        out.cursor_grab = cursor_grab;
        out.game = game_snap;

        let exit = world.request_exit;
        self.world = Some(world);
        exit
    }
}

enum ParallelCmd<G: GameApp> {
    Step {
        real_dt_ms: u64,
        inputs: InputBatch,
        spare_snapshot: RenderSnapshot<G::SnapshotData>,
    },
    Shutdown,
}

struct ParallelDone<G: GameApp> {
    snapshot: RenderSnapshot<G::SnapshotData>,
    should_exit: bool,
}

struct ParallelHost<G: GameApp> {
    cmd_tx: mpsc::Sender<ParallelCmd<G>>,
    done_rx: mpsc::Receiver<ParallelDone<G>>,
    join_handle: Option<std::thread::JoinHandle<SimRunner<G>>>,
    in_flight: bool,
}

impl<G: GameApp> ParallelHost<G> {
    fn start(
        mut sim: SimRunner<G>,
        initial_batch: InputBatch,
        spare: RenderSnapshot<G::SnapshotData>,
    ) -> Self {
        let (cmd_tx, cmd_rx) = mpsc::channel::<ParallelCmd<G>>();
        let (done_tx, done_rx) = mpsc::channel::<ParallelDone<G>>();

        let join_handle = std::thread::Builder::new()
            .name("engine-sim-parallel".to_owned())
            .spawn(
                move || {
                    while let Ok(cmd) = cmd_rx.recv() {
                        match cmd {
                            ParallelCmd::Step {
                                real_dt_ms,
                                inputs,
                                mut spare_snapshot,
                            } => {
                                let should_exit = sim.step(
                                    real_dt_ms,
                                    inputs,
                                    &mut spare_snapshot,
                                );
                                if done_tx
                                    .send(
                                        ParallelDone {
                                            snapshot: spare_snapshot,
                                            should_exit,
                                        },
                                    )
                                    .is_err()
                                {
                                    break;
                                }
                            }
                            ParallelCmd::Shutdown => break,
                        }
                    }
                    sim
                },
            )
            .unwrap_or_else(|e| panic!("Failed to spawn parallel simulation thread: {e}"));

        if let Err(e) = cmd_tx.send(
            ParallelCmd::Step {
                real_dt_ms: 16,
                inputs: initial_batch,
                spare_snapshot: spare,
            },
        ) {
            tracing::error!("Failed to send initial parallel step command: {e}");
        }

        Self {
            cmd_tx,
            done_rx,
            join_handle: Some(join_handle),
            in_flight: true,
        }
    }

    fn finish_and_stop(
        mut self,
    ) -> (
        SimRunner<G>,
        Option<ParallelDone<G>>,
    ) {
        let last = if self.in_flight {
            self.in_flight = false;
            self.done_rx.recv().ok()
        } else {
            None
        };
        if let Err(e) = self.cmd_tx.send(ParallelCmd::Shutdown) {
            tracing::warn!("Failed to send shutdown to parallel simulation thread: {e}");
        }
        let sim = self
            .join_handle
            .take()
            .expect("parallel sim handle missing")
            .join()
            .unwrap_or_else(|e| panic!("Parallel sim thread panicked: {e:?}"));
        (
            sim, last,
        )
    }
}

enum SimMode<G: GameApp> {
    Sequential(SimRunner<G>),
    Parallel(ParallelHost<G>),
    Transitioning,
}

struct RunnerApp<'a, G: GameApp, U: UiOverlay<G::SnapshotData, G::Action>> {
    config: EngineConfig,
    pending_effects: Option<EffectRegistry>,
    window: Option<Arc<Window>>,
    renderer: Option<Renderer<'a>>,
    ui: U,
    action_sender: ActionSender<G::Action>,
    sim: SimMode<G>,
    front_snapshot: RenderSnapshot<G::SnapshotData>,
    back_snapshot: RenderSnapshot<G::SnapshotData>,
    pending_inputs: InputBatch,
    last_frame: Instant,
    focused: bool,
    cursor_grabbed: bool,
}

impl<G: GameApp, U: UiOverlay<G::SnapshotData, G::Action>> RunnerApp<'_, G, U> {
    fn sync_cursor_grab(&mut self) {
        let Some(window) = self.window.as_ref() else {
            return;
        };
        let should_grab = self.focused && self.front_snapshot.cursor_grab;
        if should_grab == self.cursor_grabbed {
            return;
        }
        if should_grab {
            // Try Locked first for unbounded 360 turning; fall back to Confined where X11/compositors disallow Locked.
            if let Err(locked_err) = window.set_cursor_grab(CursorGrabMode::Locked) {
                tracing::debug!(
                    ?locked_err,
                    "CursorGrabMode::Locked unsupported; falling back to Confined"
                );
                if let Err(confined_err) = window.set_cursor_grab(CursorGrabMode::Confined) {
                    tracing::warn!(
                        ?confined_err,
                        "failed to grab cursor"
                    );
                }
            }
            window.set_cursor_visible(false);
        } else {
            if let Err(err) = window.set_cursor_grab(CursorGrabMode::None) {
                tracing::warn!(
                    ?err,
                    "failed to release cursor grab"
                );
            }
            window.set_cursor_visible(true);
        }
        self.cursor_grabbed = should_grab;
    }

    fn tick_and_render(&mut self, event_loop: &ActiveEventLoop) {
        let now = Instant::now();
        let real_dt_ms = now
            .saturating_duration_since(self.last_frame)
            .as_millis()
            .clamp(
                1, 50,
            ) as u64;
        self.last_frame = now;

        let batch = std::mem::take(&mut self.pending_inputs);
        let mut should_exit = false;

        match std::mem::replace(
            &mut self.sim,
            SimMode::Transitioning,
        ) {
            SimMode::Sequential(mut sim) => {
                should_exit = sim.step(
                    real_dt_ms,
                    batch,
                    &mut self.front_snapshot,
                );

                self.ui.sync(
                    &self.front_snapshot,
                    &self.action_sender,
                );
                if let Some(renderer) = self.renderer.as_mut() {
                    renderer.draw(
                        &self.front_snapshot,
                        &mut self.ui,
                    );
                }

                if self.front_snapshot.schedule == RenderSchedule::Parallel {
                    let spare = std::mem::take(&mut self.back_snapshot);
                    let host = ParallelHost::start(
                        sim,
                        InputBatch::default(),
                        spare,
                    );
                    self.sim = SimMode::Parallel(host);
                } else {
                    self.sim = SimMode::Sequential(sim);
                }
            }
            SimMode::Parallel(mut host) => {
                // Collect completed `Sim` frame into `front_snapshot` and immediately start `Sim N+1`
                // while main thread renders `front_snapshot`.
                if host.in_flight
                    && let Ok(done) = host.done_rx.recv()
                {
                    should_exit = done.should_exit;
                    let old_front = std::mem::replace(
                        &mut self.front_snapshot,
                        done.snapshot,
                    );
                    self.back_snapshot = old_front;
                    host.in_flight = false;
                }

                let desired_schedule = self.front_snapshot.schedule;
                if desired_schedule == RenderSchedule::Sequential {
                    let (mut sim, extra) = host.finish_and_stop();
                    if let Some(done) = extra {
                        should_exit |= done.should_exit;
                        self.front_snapshot = done.snapshot;
                    }
                    // Run current batch synchronously so switching back to Sequential drops lag immediately.
                    if !batch.is_empty() {
                        should_exit |= sim.step(
                            real_dt_ms,
                            batch,
                            &mut self.front_snapshot,
                        );
                    }
                    self.ui.sync(
                        &self.front_snapshot,
                        &self.action_sender,
                    );
                    if let Some(renderer) = self.renderer.as_mut() {
                        renderer.draw(
                            &self.front_snapshot,
                            &mut self.ui,
                        );
                    }
                    self.sim = SimMode::Sequential(sim);
                } else {
                    let spare = std::mem::take(&mut self.back_snapshot);
                    if host
                        .cmd_tx
                        .send(
                            ParallelCmd::Step {
                                real_dt_ms,
                                inputs: batch,
                                spare_snapshot: spare,
                            },
                        )
                        .is_ok()
                    {
                        host.in_flight = true;
                    }

                    self.ui.sync(
                        &self.front_snapshot,
                        &self.action_sender,
                    );
                    if let Some(renderer) = self.renderer.as_mut() {
                        renderer.draw(
                            &self.front_snapshot,
                            &mut self.ui,
                        );
                    }

                    self.sim = SimMode::Parallel(host);
                }
            }
            SimMode::Transitioning => unreachable!("sim state transition interrupted"),
        }

        self.sync_cursor_grab();
        if should_exit {
            event_loop.exit();
        }
    }
}

impl<G: GameApp, U: UiOverlay<G::SnapshotData, G::Action>> ApplicationHandler
    for RunnerApp<'_, G, U>
{
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let attrs = Window::default_attributes()
            .with_title(self.config.title.clone())
            .with_inner_size(
                winit::dpi::LogicalSize::new(
                    self.config.width,
                    self.config.height,
                ),
            );

        let window = Arc::new(
            event_loop
                .create_window(attrs)
                .unwrap_or_else(|e| panic!("Failed to create window: {e}")),
        );
        let size = window.inner_size();
        let effects = self.pending_effects.take().unwrap_or_default();
        let renderer = Renderer::new(
            Arc::clone(&window),
            size,
            effects,
        );

        self.window = Some(window);
        self.renderer = Some(renderer);
        self.focused = true;
        self.sync_cursor_grab();
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        if let WindowEvent::CloseRequested = event {
            event_loop.exit();
            return;
        }

        if let WindowEvent::Focused(focused) = event {
            self.focused = focused;
            self.sync_cursor_grab();
        }

        if let WindowEvent::Resized(size) = event
            && let Some(renderer) = self.renderer.as_mut()
        {
            renderer.resize(size);
        }

        let consumed_by_ui = if let Some(window) = self.window.as_ref() {
            self.ui.handle_event(
                &event, window,
            )
        } else {
            false
        };

        match event {
            WindowEvent::RedrawRequested => {
                self.tick_and_render(event_loop);
            }
            WindowEvent::KeyboardInput { event: key_ev, .. } if !consumed_by_ui => {
                if let PhysicalKey::Code(code) = key_ev.physical_key {
                    self.pending_inputs.key_events.push(
                        KeyTransition {
                            key: code,
                            state: key_ev.state,
                            repeat: key_ev.repeat,
                        },
                    );
                }
            }
            WindowEvent::MouseInput { state, button, .. } if !consumed_by_ui => {
                self.pending_inputs
                    .mouse_buttons
                    .push(MouseTransition { button, state });
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.pending_inputs.cursor_pos = Some(
                    glam::Vec2::new(
                        position.x as f32,
                        position.y as f32,
                    ),
                );
            }
            _ => {}
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: DeviceId,
        event: DeviceEvent,
    ) {
        if self.focused
            && let DeviceEvent::MouseMotion { delta: (dx, dy) } = event
        {
            self.pending_inputs.mouse_delta += glam::Vec2::new(
                dx as f32, dy as f32,
            );
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(window) = self.window.as_ref() {
            window.request_redraw();
        }
    }

    fn exiting(&mut self, _event_loop: &ActiveEventLoop) {
        if let SimMode::Parallel(host) = std::mem::replace(
            &mut self.sim,
            SimMode::Transitioning,
        ) {
            let (_sim, _) = host.finish_and_stop();
        }
    }
}

/// Top-level runtime host owning the main loop, `RenderSnapshot` buffers, and `SimRunner`.
pub struct EngineRunner;

impl EngineRunner {
    /// Starts the `winit` event loop and drives `GameApp` and `UiOverlay` with no extra effects.
    pub fn run<G, U>(config: EngineConfig, game: G, ui: U) -> Result<(), EventLoopError>
    where
        G: GameApp,
        U: UiOverlay<G::SnapshotData, G::Action>,
    {
        Self::run_with_effects(
            config,
            EffectRegistry::new(),
            game,
            ui,
        )
    }

    /// Starts the `winit` event loop with registered pluggable `RenderEffect` crates.
    pub fn run_with_effects<G, U>(
        config: EngineConfig,
        effects: EffectRegistry,
        game: G,
        ui: U,
    ) -> Result<(), EventLoopError>
    where
        G: GameApp,
        U: UiOverlay<G::SnapshotData, G::Action>,
    {
        let sim_runner = SimRunner::new(
            game,
            config.initial_schedule,
        );
        let action_sender = sim_runner
            .world
            .as_ref()
            .expect("world initialized")
            .action_sender
            .clone();
        let front = RenderSnapshot {
            schedule: config.initial_schedule,
            ..Default::default()
        };
        let mut back = RenderSnapshot {
            schedule: config.initial_schedule,
            ..Default::default()
        };

        let sim = match config.initial_schedule {
            RenderSchedule::Sequential => SimMode::Sequential(sim_runner),
            RenderSchedule::Parallel => {
                let spare = std::mem::take(&mut back);
                SimMode::Parallel(
                    ParallelHost::start(
                        sim_runner,
                        InputBatch::default(),
                        spare,
                    ),
                )
            }
        };

        let event_loop = EventLoop::new()?;
        event_loop.set_control_flow(ControlFlow::Poll);

        let mut app = RunnerApp {
            config,
            pending_effects: Some(effects),
            window: None,
            renderer: None,
            ui,
            action_sender,
            sim,
            front_snapshot: front,
            back_snapshot: back,
            pending_inputs: InputBatch::default(),
            last_frame: Instant::now(),
            focused: false,
            cursor_grabbed: false,
        };

        event_loop.run_app(&mut app)
    }
}
