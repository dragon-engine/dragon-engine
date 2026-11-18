use std::any::Any;
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::task::Waker;

use slotmap::SlotMap;

use crate::clock::{ClockDomain, DualClocks};
use crate::effect::EffectAttachment;
use crate::executor::{ComputeTask, SimAccess, TaskHandle, WorkerPool, with_sim_mut};
use crate::modes::ModeManager;
use crate::physics::{ActorId, BodyKind, PhysicsCommand, PhysicsWorld, SpatialQuery};
use crate::runner::{GameApp, RenderSchedule};
use crate::save::ActorCheckpoint;
use crate::snapshot::{CameraSnapshot, RenderMesh};
use crate::sync::{ActionReceiver, ActionSender, SleepFuture, WaitError, WaitTimeout};

/// Spawn parameters for creating an actor with synchronized AoS, SoA, and Rapier3D state.
#[derive(Debug, Clone)]
pub struct ActorSpawnDesc<A> {
    pub position: glam::Vec3,
    pub rotation: glam::Quat,
    pub half_extents: glam::Vec3,
    pub color: [f32; 3],
    pub effect: Option<EffectAttachment>,
    pub body_kind: BodyKind,
    pub mass: f32,
    pub data: A,
}

pub(crate) struct PendingTaskSpawn {
    pub future: Pin<Box<dyn Future<Output = ()> + Send>>,
    pub cancelled: Arc<AtomicBool>,
}

impl PendingTaskSpawn {
    pub(crate) fn is_cancelled(&self) -> bool {
        self.cancelled.load(std::sync::atomic::Ordering::Acquire)
    }
}

struct ActorRecord<A> {
    data: A,
    tasks: Vec<TaskHandle>,
    buffs: HashMap<u64, TaskHandle>,
}

/// Read-only view of engine world state exposed to `GameApp::build_snapshot`.
pub struct WorldReadView<'a, G: GameApp> {
    world: &'a WorldState<G>,
}

impl<G: GameApp> WorldReadView<'_, G> {
    pub fn read_actor<R>(&self, id: ActorId, f: impl FnOnce(&G::ActorData) -> R) -> Option<R> {
        self.world.actors.get(id).map(|rec| f(&rec.data))
    }

    pub fn actor_position(&self, id: ActorId) -> Option<glam::Vec3> {
        self.world.physics.actor_position(id)
    }

    pub fn actor_rotation(&self, id: ActorId) -> Option<glam::Quat> {
        self.world
            .physics
            .soa
            .index_of(id)
            .map(|idx| self.world.physics.soa.rotations[idx])
    }

    pub fn modes(&self) -> &ModeManager<G::FlatMode, G::StackMode> {
        &self.world.modes
    }

    pub fn world_time_ms(&self) -> u64 {
        self.world.clocks.world_time_ms()
    }

    pub fn ui_time_ms(&self) -> u64 {
        self.world.clocks.ui_time_ms()
    }

    pub fn collect_actor_checkpoints(&self) -> Vec<ActorCheckpoint<G::ActorData>> {
        self.world.collect_actor_checkpoints()
    }
}

/// Private simulation state holding `SlotMap`, `PhysicsWorld`, `DualClocks`, `ModeManager`, and `WorkerPool`.
pub(crate) struct WorldState<G: GameApp> {
    actors: SlotMap<ActorId, ActorRecord<G::ActorData>>,
    pub global_tasks: Vec<TaskHandle>,
    pub physics: PhysicsWorld,
    pub clocks: DualClocks,
    pub modes: ModeManager<G::FlatMode, G::StackMode>,
    pub pending_physics_cmds: Vec<PhysicsCommand>,
    pub pending_spawns: Vec<PendingTaskSpawn>,
    pub worker_pool: WorkerPool,
    pub schedule: RenderSchedule,
    pub action_sender: ActionSender<G::Action>,
    pub action_receiver: ActionReceiver<G::Action>,
    pub request_exit: bool,
}

impl<G: GameApp> WorldState<G> {
    pub(crate) fn new(
        action_sender: ActionSender<G::Action>,
        action_receiver: ActionReceiver<G::Action>,
    ) -> Self {
        let (initial_flat, root_stack, world_active) = G::initial_modes();
        Self {
            actors: SlotMap::with_key(),
            global_tasks: Vec::new(),
            physics: PhysicsWorld::new(),
            clocks: DualClocks::new(),
            modes: ModeManager::new(
                initial_flat,
                root_stack,
                world_active,
            ),
            pending_physics_cmds: Vec::new(),
            pending_spawns: Vec::new(),
            worker_pool: WorkerPool::new(2),
            schedule: RenderSchedule::Sequential,
            action_sender,
            action_receiver,
            request_exit: false,
        }
    }

    fn enqueue_spawn(&mut self, future: impl Future<Output = ()> + Send + 'static) -> TaskHandle {
        let cancelled = Arc::new(AtomicBool::new(false));
        self.pending_spawns.push(
            PendingTaskSpawn {
                future: Box::pin(future),
                cancelled: Arc::clone(&cancelled),
            },
        );
        TaskHandle::new(cancelled)
    }

    pub(crate) fn spawn_actor(&mut self, desc: ActorSpawnDesc<G::ActorData>) -> ActorId {
        let id = self.actors.insert(
            ActorRecord {
                data: desc.data,
                tasks: Vec::new(),
                buffs: HashMap::new(),
            },
        );
        self.physics.spawn_body(
            id,
            desc.position,
            desc.rotation,
            desc.half_extents,
            desc.color,
            desc.effect,
            desc.body_kind,
            desc.mass,
        );
        id
    }

    pub(crate) fn despawn_actor(&mut self, id: ActorId) -> bool {
        // Removing from `SlotMap` drops all `TaskHandle`s stored in `ActorRecord` (`tasks` and `buffs`).
        if self.actors.remove(id).is_some() {
            self.physics.despawn_body(id);
            true
        } else {
            false
        }
    }

    pub(crate) fn read_actor<R>(
        &self,
        id: ActorId,
        f: impl FnOnce(&G::ActorData) -> R,
    ) -> Option<R> {
        self.actors.get(id).map(|rec| f(&rec.data))
    }

    pub(crate) fn write_actor<R>(
        &mut self,
        id: ActorId,
        f: impl FnOnce(&mut G::ActorData) -> R,
    ) -> Option<R> {
        self.actors.get_mut(id).map(|rec| f(&mut rec.data))
    }

    pub(crate) fn attach_actor_script<Fut>(
        &mut self,
        id: ActorId,
        f: impl FnOnce(ScriptContext<G>) -> Fut,
    ) -> bool
    where
        Fut: Future<Output = ()> + Send + 'static,
    {
        if !self.actors.contains_key(id) {
            return false;
        }
        let ctx = self.script_context();
        let handle = self.enqueue_spawn(f(ctx));
        if let Some(rec) = self.actors.get_mut(id) {
            rec.tasks.push(handle);
            true
        } else {
            false
        }
    }

    pub(crate) fn spawn_buff<Fut>(
        &mut self,
        id: ActorId,
        buff_key: u64,
        f: impl FnOnce(ScriptContext<G>) -> Fut,
    ) -> bool
    where
        Fut: Future<Output = ()> + Send + 'static,
    {
        if !self.actors.contains_key(id) {
            return false;
        }
        let ctx = self.script_context();
        let handle = self.enqueue_spawn(f(ctx));
        if let Some(rec) = self.actors.get_mut(id) {
            // Replacing the old `TaskHandle` cancels the previous buff script on refresh.
            rec.buffs.insert(
                buff_key, handle,
            );
            true
        } else {
            false
        }
    }

    pub(crate) fn spawn_global_script<Fut>(&mut self, f: impl FnOnce(ScriptContext<G>) -> Fut)
    where
        Fut: Future<Output = ()> + Send + 'static,
    {
        let ctx = self.script_context();
        let handle = self.enqueue_spawn(f(ctx));
        self.global_tasks.push(handle);
    }

    pub(crate) fn push_mode_with_script<Fut>(
        &mut self,
        mode: G::StackMode,
        f: impl FnOnce(ScriptContext<G>) -> Fut,
    ) where
        Fut: Future<Output = ()> + Send + 'static,
    {
        let ctx = self.script_context();
        let handle = self.enqueue_spawn(f(ctx));
        self.modes.push_with_handle(
            mode, handle,
        );
    }

    pub(crate) fn script_context(&self) -> ScriptContext<G> {
        ScriptContext {
            action_sender: self.action_sender.clone(),
            action_receiver: self.action_receiver.clone(),
        }
    }

    pub(crate) fn read_view(&self) -> WorldReadView<'_, G> {
        WorldReadView { world: self }
    }

    pub(crate) fn collect_render_meshes(&self, camera: &CameraSnapshot) -> Vec<RenderMesh> {
        // Use octree culling around camera target to skip far off-map meshes when large worlds exist.
        let cull_region = crate::physics::Aabb3::from_center_half_extents(
            camera.target,
            glam::Vec3::splat(200.0),
        );
        let mut visible_indices = Vec::new();
        self.physics.octree.query(
            &cull_region,
            &mut visible_indices,
        );

        let mut out = Vec::with_capacity(visible_indices.len());
        for idx in visible_indices {
            let mesh = self.physics.soa.meshes[idx];
            if !mesh.visible {
                continue;
            }
            out.push(
                RenderMesh {
                    actor: self.physics.soa.actor_ids[idx],
                    position: self.physics.soa.positions[idx],
                    rotation: self.physics.soa.rotations[idx],
                    half_extents: mesh.half_extents,
                    color: mesh.color,
                    effect: mesh.effect,
                },
            );
        }
        out
    }

    pub(crate) fn collect_actor_checkpoints(&self) -> Vec<ActorCheckpoint<G::ActorData>> {
        let mut out = Vec::with_capacity(self.physics.soa.actor_ids.len());
        for (saved_id, &actor_id) in self.physics.soa.actor_ids.iter().enumerate() {
            if let Some(rec) = self.actors.get(actor_id) {
                let pos = self.physics.soa.positions[saved_id];
                let rot = self.physics.soa.rotations[saved_id];
                let half = self.physics.soa.half_extents[saved_id];
                let mesh = self.physics.soa.meshes[saved_id];
                let body_kind = self.physics.soa.body_kinds[saved_id];
                out.push(
                    ActorCheckpoint {
                        saved_id,
                        position: pos.to_array(),
                        rotation: rot.to_array(),
                        half_extents: half.to_array(),
                        color: mesh.color,
                        effect: mesh.effect,
                        body_kind,
                        mass: 1.0,
                        data: rec.data.clone(),
                    },
                );
            }
        }
        out
    }

    pub(crate) fn clear_world_for_restore(&mut self, world_time_ms: u64) {
        let ids: Vec<ActorId> = self.actors.keys().collect();
        for id in ids {
            self.despawn_actor(id);
        }
        self.global_tasks.clear();
        self.pending_physics_cmds.clear();
        self.pending_spawns.clear();
        self.clocks.set_world_time_ms(world_time_ms);
    }
}

impl<G: GameApp> SimAccess for WorldState<G> {
    fn domain_time_ms(&self, domain: ClockDomain) -> u64 {
        self.clocks.domain_time_ms(domain)
    }

    fn register_timer(&mut self, domain: ClockDomain, deadline_ms: u64, waker: Waker) {
        self.clocks.register_timer(
            domain,
            deadline_ms,
            waker,
        );
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn into_any(self: Box<Self>) -> Box<dyn Any> {
        self
    }
}

/// Initialization context passed to `GameApp::init`.
pub struct EngineContext<'a, G: GameApp> {
    pub(crate) world: &'a mut WorldState<G>,
}

impl<G: GameApp> EngineContext<'_, G> {
    pub fn spawn_actor(&mut self, desc: ActorSpawnDesc<G::ActorData>) -> ActorId {
        self.world.spawn_actor(desc)
    }

    pub fn attach_actor_script<Fut>(
        &mut self,
        id: ActorId,
        f: impl FnOnce(ScriptContext<G>) -> Fut,
    ) -> bool
    where
        Fut: Future<Output = ()> + Send + 'static,
    {
        self.world.attach_actor_script(
            id, f,
        )
    }

    pub fn spawn_global_script<Fut>(&mut self, f: impl FnOnce(ScriptContext<G>) -> Fut)
    where
        Fut: Future<Output = ()> + Send + 'static,
    {
        self.world.spawn_global_script(f);
    }

    pub fn action_sender(&self) -> ActionSender<G::Action> {
        self.world.action_sender.clone()
    }

    pub fn set_render_schedule(&mut self, schedule: RenderSchedule) {
        self.world.schedule = schedule;
    }
}

/// Input handling context passed to `GameApp::on_input`.
pub struct InputContext<'a, G: GameApp> {
    pub(crate) world: &'a mut WorldState<G>,
}

impl<G: GameApp> InputContext<'_, G> {
    pub fn modes(&self) -> &ModeManager<G::FlatMode, G::StackMode> {
        &self.world.modes
    }

    pub fn set_flat_mode(&mut self, flat: G::FlatMode) {
        self.world.modes.set_flat(flat);
    }

    pub fn push_mode(&mut self, mode: G::StackMode) {
        self.world.modes.push(mode);
    }

    pub fn push_mode_with_script<Fut>(
        &mut self,
        mode: G::StackMode,
        f: impl FnOnce(ScriptContext<G>) -> Fut,
    ) where
        Fut: Future<Output = ()> + Send + 'static,
    {
        self.world.push_mode_with_script(
            mode, f,
        );
    }

    pub fn pop_mode(&mut self) -> Option<G::StackMode> {
        self.world.modes.pop()
    }

    pub fn action_sender(&self) -> &ActionSender<G::Action> {
        &self.world.action_sender
    }

    pub fn is_world_active(&self) -> bool {
        self.world.modes.is_world_active()
    }

    pub fn set_render_schedule(&mut self, schedule: RenderSchedule) {
        self.world.schedule = schedule;
    }

    pub fn request_exit(&mut self) {
        self.world.request_exit = true;
    }
}

/// Synchronous post-step context used for OS/disk operations (like checkpoint save/load)
/// strictly outside async gameplay futures.
pub struct SyncContext<'a, G: GameApp> {
    pub(crate) world: &'a mut WorldState<G>,
}

impl<G: GameApp> SyncContext<'_, G> {
    pub fn world_time_ms(&self) -> u64 {
        self.world.clocks.world_time_ms()
    }

    pub fn pause_world_for_checkpoint(&mut self) {
        self.world.clocks.pause_world_for_checkpoint();
    }

    pub fn resume_world_from_checkpoint(&mut self) {
        self.world.clocks.resume_world_from_checkpoint();
    }

    pub fn collect_actor_checkpoints(&self) -> Vec<ActorCheckpoint<G::ActorData>> {
        self.world.collect_actor_checkpoints()
    }

    pub fn restore_checkpoint_actors(
        &mut self,
        world_time_ms: u64,
        checkpoints: Vec<ActorCheckpoint<G::ActorData>>,
    ) -> Vec<ActorId> {
        self.world.clear_world_for_restore(world_time_ms);
        let mut ids = Vec::with_capacity(checkpoints.len());
        for cp in checkpoints {
            let id = self.world.spawn_actor(
                ActorSpawnDesc {
                    position: glam::Vec3::from_array(cp.position),
                    rotation: glam::Quat::from_array(cp.rotation),
                    half_extents: glam::Vec3::from_array(cp.half_extents),
                    color: cp.color,
                    effect: cp.effect,
                    body_kind: cp.body_kind,
                    mass: cp.mass,
                    data: cp.data,
                },
            );
            ids.push(id);
        }
        ids
    }

    pub fn attach_actor_script<Fut>(
        &mut self,
        id: ActorId,
        f: impl FnOnce(ScriptContext<G>) -> Fut,
    ) -> bool
    where
        Fut: Future<Output = ()> + Send + 'static,
    {
        self.world.attach_actor_script(
            id, f,
        )
    }

    pub fn spawn_buff<Fut>(
        &mut self,
        id: ActorId,
        buff_key: u64,
        f: impl FnOnce(ScriptContext<G>) -> Fut,
    ) -> bool
    where
        Fut: Future<Output = ()> + Send + 'static,
    {
        self.world.spawn_buff(
            id, buff_key, f,
        )
    }

    pub fn spawn_global_script<Fut>(&mut self, f: impl FnOnce(ScriptContext<G>) -> Fut)
    where
        Fut: Future<Output = ()> + Send + 'static,
    {
        self.world.spawn_global_script(f);
    }

    pub fn read_actor<R>(&self, id: ActorId, f: impl FnOnce(&G::ActorData) -> R) -> Option<R> {
        self.world.read_actor(
            id, f,
        )
    }

    pub fn set_flat_mode(&mut self, flat: G::FlatMode) {
        self.world.modes.set_flat(flat);
    }
}

/// Handle passed to `async` gameplay scripts running in `TaskSlab`.
/// All storage accessors run synchronous closures so borrows can never be held across `.await`.
pub struct ScriptContext<G: GameApp> {
    action_sender: ActionSender<G::Action>,
    action_receiver: ActionReceiver<G::Action>,
}

impl<G: GameApp> Clone for ScriptContext<G> {
    fn clone(&self) -> Self {
        Self {
            action_sender: self.action_sender.clone(),
            action_receiver: self.action_receiver.clone(),
        }
    }
}

impl<G: GameApp> ScriptContext<G> {
    pub fn actions(&self) -> &ActionSender<G::Action> {
        &self.action_sender
    }

    pub fn receiver(&self) -> &ActionReceiver<G::Action> {
        &self.action_receiver
    }

    /// Awaits a duration governed by `WaitTimeout` (`WorldMs` or `UiMs`).
    pub async fn sleep(&self, timeout: WaitTimeout) -> Result<(), WaitError> {
        match SleepFuture::from_timeout(timeout) {
            Some(fut) => fut.await,
            None => Ok(()),
        }
    }

    pub fn world_time_ms(&self) -> u64 {
        with_sim_mut::<WorldState<G>, u64>(|w| w.clocks.world_time_ms()).unwrap_or(0)
    }

    pub fn ui_time_ms(&self) -> u64 {
        with_sim_mut::<WorldState<G>, u64>(|w| w.clocks.ui_time_ms()).unwrap_or(0)
    }

    pub fn is_world_active(&self) -> bool {
        with_sim_mut::<WorldState<G>, bool>(|w| w.modes.is_world_active()).unwrap_or(false)
    }

    pub fn flat_mode(&self) -> Option<G::FlatMode> {
        with_sim_mut::<WorldState<G>, G::FlatMode>(|w| w.modes.flat().clone())
    }

    pub fn stack_top(&self) -> Option<G::StackMode> {
        with_sim_mut::<WorldState<G>, Option<G::StackMode>>(|w| w.modes.stack_top().cloned())
            .flatten()
    }

    pub fn set_flat_mode(&self, flat: G::FlatMode) {
        with_sim_mut::<WorldState<G>, ()>(|w| w.modes.set_flat(flat));
    }

    pub fn push_mode(&self, mode: G::StackMode) {
        with_sim_mut::<WorldState<G>, ()>(|w| w.modes.push(mode));
    }

    pub fn push_mode_with_script<Fut>(
        &self,
        mode: G::StackMode,
        f: impl FnOnce(ScriptContext<G>) -> Fut,
    ) where
        Fut: Future<Output = ()> + Send + 'static,
    {
        with_sim_mut::<WorldState<G>, ()>(
            |w| {
                w.push_mode_with_script(
                    mode, f,
                )
            },
        );
    }

    pub fn pop_mode(&self) -> Option<G::StackMode> {
        with_sim_mut::<WorldState<G>, Option<G::StackMode>>(|w| w.modes.pop()).flatten()
    }

    pub fn set_render_schedule(&self, schedule: RenderSchedule) {
        with_sim_mut::<WorldState<G>, ()>(|w| w.schedule = schedule);
    }

    pub fn request_exit(&self) {
        with_sim_mut::<WorldState<G>, ()>(|w| w.request_exit = true);
    }

    pub fn spawn_actor(&self, desc: ActorSpawnDesc<G::ActorData>) -> Option<ActorId> {
        with_sim_mut::<WorldState<G>, ActorId>(|w| w.spawn_actor(desc))
    }

    pub fn despawn_actor(&self, id: ActorId) -> bool {
        with_sim_mut::<WorldState<G>, bool>(|w| w.despawn_actor(id)).unwrap_or(false)
    }

    pub fn read_actor<R>(&self, id: ActorId, f: impl FnOnce(&G::ActorData) -> R) -> Option<R> {
        with_sim_mut::<WorldState<G>, Option<R>>(
            |w| {
                w.read_actor(
                    id, f,
                )
            },
        )
        .flatten()
    }

    pub fn write_actor<R>(&self, id: ActorId, f: impl FnOnce(&mut G::ActorData) -> R) -> Option<R> {
        with_sim_mut::<WorldState<G>, Option<R>>(
            |w| {
                w.write_actor(
                    id, f,
                )
            },
        )
        .flatten()
    }

    pub fn attach_actor_script<Fut>(
        &self,
        id: ActorId,
        f: impl FnOnce(ScriptContext<G>) -> Fut,
    ) -> bool
    where
        Fut: Future<Output = ()> + Send + 'static,
    {
        with_sim_mut::<WorldState<G>, bool>(
            |w| {
                w.attach_actor_script(
                    id, f,
                )
            },
        )
        .unwrap_or(false)
    }

    /// Spawns or refreshes a keyed buff script on an actor. Existing buff scripts with the same key cancel automatically.
    pub fn spawn_buff<Fut>(
        &self,
        id: ActorId,
        buff_key: u64,
        f: impl FnOnce(ScriptContext<G>) -> Fut,
    ) -> bool
    where
        Fut: Future<Output = ()> + Send + 'static,
    {
        with_sim_mut::<WorldState<G>, bool>(
            |w| {
                w.spawn_buff(
                    id, buff_key, f,
                )
            },
        )
        .unwrap_or(false)
    }

    pub fn spawn_global_script<Fut>(&self, f: impl FnOnce(ScriptContext<G>) -> Fut)
    where
        Fut: Future<Output = ()> + Send + 'static,
    {
        with_sim_mut::<WorldState<G>, ()>(|w| w.spawn_global_script(f));
    }

    pub fn spatial<R>(&self, f: impl FnOnce(&dyn SpatialQuery) -> R) -> Option<R> {
        with_sim_mut::<WorldState<G>, R>(|w| f(&w.physics))
    }

    pub fn push_physics_command(&self, cmd: PhysicsCommand) {
        with_sim_mut::<WorldState<G>, ()>(|w| w.pending_physics_cmds.push(cmd));
    }

    /// Dispatches pure compute onto the background `async-executor` worker pool using a cloned input snapshot.
    pub fn spawn_compute<S, T, Fut>(
        &self,
        snapshot: S,
        f: impl FnOnce(S) -> Fut + Send + 'static,
    ) -> Option<ComputeTask<T>>
    where
        S: Send + 'static,
        T: Send + 'static,
        Fut: Future<Output = T> + Send + 'static,
    {
        with_sim_mut::<WorldState<G>, ComputeTask<T>>(|w| w.worker_pool.spawn(f(snapshot)))
    }

    pub fn pause_world_for_checkpoint(&self) {
        with_sim_mut::<WorldState<G>, ()>(|w| w.clocks.pause_world_for_checkpoint());
    }

    pub fn resume_world_from_checkpoint(&self) {
        with_sim_mut::<WorldState<G>, ()>(|w| w.clocks.resume_world_from_checkpoint());
    }

    pub fn collect_actor_checkpoints(&self) -> Vec<ActorCheckpoint<G::ActorData>> {
        with_sim_mut::<WorldState<G>, Vec<ActorCheckpoint<G::ActorData>>>(
            |w| w.collect_actor_checkpoints(),
        )
        .unwrap_or_default()
    }

    /// Clears existing entities and scripts, restores actors and world time from checkpoint data.
    pub fn restore_checkpoint_actors(
        &self,
        world_time_ms: u64,
        checkpoints: Vec<ActorCheckpoint<G::ActorData>>,
    ) -> Vec<ActorId> {
        with_sim_mut::<WorldState<G>, Vec<ActorId>>(
            |w| {
                w.clear_world_for_restore(world_time_ms);
                let mut ids = Vec::with_capacity(checkpoints.len());
                for cp in checkpoints {
                    let id = w.spawn_actor(
                        ActorSpawnDesc {
                            position: glam::Vec3::from_array(cp.position),
                            rotation: glam::Quat::from_array(cp.rotation),
                            half_extents: glam::Vec3::from_array(cp.half_extents),
                            color: cp.color,
                            effect: cp.effect,
                            body_kind: cp.body_kind,
                            mass: cp.mass,
                            data: cp.data,
                        },
                    );
                    ids.push(id);
                }
                ids
            },
        )
        .unwrap_or_default()
    }
}
