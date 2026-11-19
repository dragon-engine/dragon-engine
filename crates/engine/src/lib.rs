//! Core runtime crate enforcing engine invariants through private storage and typed APIs.

mod clock;
mod effect;
mod executor;
mod gpu;
mod input;
mod modes;
mod physics;
mod renderer;
mod runner;
mod save;
mod snapshot;
mod sync;
mod ui;
mod world;

pub use clock::ClockDomain;
pub use effect::{
    EffectAttachment, EffectBlendMode, EffectId, EffectRegistry, EffectShaderSpec, RenderEffect,
};
pub use executor::ComputeTask;
pub use futures::{join, select_biased};
pub use gpu::GpuError;
pub use input::{GameInput, KeyTransition, MouseTransition};
pub use modes::ModeManager;
pub use physics::{Aabb3, ActorId, BodyKind, PhysicsCommand, RaycastHit, SpatialQuery};
pub use runner::{EngineConfig, EngineError, EngineRunner, GameApp, RenderSchedule};
pub use save::{
    ActorCheckpoint, SaveCodec, TimerCheckpoint, WorldCheckpoint, read_checkpoint_file,
    write_checkpoint_file,
};
pub use shader_abi;
pub use snapshot::{CameraSnapshot, GizmoBox, RenderMesh, RenderSnapshot};
pub use sync::{ActionReceiver, ActionSender, StateSignal, WaitError, WaitTimeout};
pub use ui::{UiBlitter, UiOverlay, UiRenderTarget};
pub use winit;

/// Validates an external `RenderEffect` SPIR-V shader against `engine`'s Vulkan 1.3 pipeline ABI.
pub fn validate_effect_shader_vulkan(spec: EffectShaderSpec) -> Result<(), GpuError> {
    gpu::scene_pipeline::validate_effect_shader_headless(spec)
}

pub fn init_tracing() {
    if let Err(e) = tracing_subscriber::fmt::try_init() {
        tracing::debug!("Tracing subscriber already initialized: {e}");
    }
}

pub use world::{
    ActorSpawnDesc, EngineContext, InputContext, ScriptContext, SyncContext, WorldReadView,
};

#[cfg(test)]
#[allow(clippy::disallowed_methods)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;
    use crate::executor::TaskSlab;
    use crate::input::InputBatch;
    use crate::world::WorldState;

    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    enum TestFlat {
        Menu,
        Game,
    }

    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    enum TestStack {
        Play,
        Inv,
    }

    #[derive(Clone, Debug, Default)]
    struct TestActor {
        hp: i32,
    }

    struct TestApp;

    impl GameApp for TestApp {
        type Action = u32;
        type ActorData = TestActor;
        type FlatMode = TestFlat;
        type SnapshotData = ();
        type StackMode = TestStack;

        fn initial_modes() -> (
            Self::FlatMode,
            Self::StackMode,
            fn(&Self::FlatMode, &[Self::StackMode]) -> bool,
        ) {
            (
                TestFlat::Game,
                TestStack::Play,
                |f, s| *f == TestFlat::Game && s.last() == Some(&TestStack::Play),
            )
        }

        fn init(&mut self, _ctx: &mut EngineContext<Self>) {}

        fn world_scale(&self) -> f32 {
            1.0
        }

        fn on_input(&mut self, _input: GameInput, _ctx: &mut InputContext<Self>) {}

        fn build_snapshot(
            &self,
            _world: &WorldReadView<Self>,
        ) -> (
            CameraSnapshot,
            Self::SnapshotData,
        ) {
            (
                CameraSnapshot::default(),
                (),
            )
        }
    }

    fn drain<G: GameApp>(world: &mut WorldState<G>, slab: &mut TaskSlab) {
        for p in world.pending_spawns.drain(..) {
            if !p.is_cancelled() {
                let h = slab.spawn_with_token(
                    p.future,
                    p.cancelled,
                );
                std::mem::forget(h);
            }
        }
    }

    #[test]
    fn hybrid_modes_preserve_stack_and_cancel_mode_scripts_on_pop() {
        let (tx, rx) = sync::action_channel(16);
        let mut world = Box::new(
            WorldState::<TestApp>::new(
                tx, rx,
            ),
        );
        let mut slab = TaskSlab::new();
        let drops = Arc::new(AtomicUsize::new(0));
        let drops_clone = Arc::clone(&drops);

        struct DropCounter(Arc<AtomicUsize>);
        impl Drop for DropCounter {
            fn drop(&mut self) {
                self.0.fetch_add(
                    1,
                    Ordering::SeqCst,
                );
            }
        }

        world.push_mode_with_script(
            TestStack::Inv,
            move |ctx| async move {
                let _guard = DropCounter(drops_clone);
                if let Err(e) = ctx.sleep(WaitTimeout::UiMs(10_000)).await {
                    tracing::error!("Test sleep failed: {e}");
                }
            },
        );
        drain(
            &mut world, &mut slab,
        );
        let mut world = slab.poll_ready(world);
        assert_eq!(
            drops.load(Ordering::SeqCst),
            0
        );

        // Toggle flat mode without clearing StackMode stack.
        world.modes.set_flat(TestFlat::Menu);
        assert_eq!(
            world.modes.stack_modes(),
            vec![TestStack::Play, TestStack::Inv]
        );
        assert!(!world.modes.is_world_active());

        // Pop Inventory -> ModeLayer drops -> mode script cancels on next poll.
        assert_eq!(
            world.modes.pop(),
            Some(TestStack::Inv)
        );
        let _world = slab.poll_ready(world);
        assert_eq!(
            drops.load(Ordering::SeqCst),
            1
        );
    }

    #[test]
    fn dual_clocks_actor_despawn_and_buff_refresh() {
        let (tx, rx) = sync::action_channel(16);
        let mut world = Box::new(
            WorldState::<TestApp>::new(
                tx, rx,
            ),
        );
        let mut slab = TaskSlab::new();

        let id = world.spawn_actor(
            ActorSpawnDesc {
                position: glam::Vec3::new(
                    0.0, 2.0, 0.0,
                ),
                rotation: glam::Quat::IDENTITY,
                half_extents: glam::Vec3::splat(0.5),
                color: [1.0, 0.0, 0.0],
                effect: None,
                body_kind: BodyKind::Dynamic,
                mass: 1.0,
                data: TestActor { hp: 50 },
            },
        );

        let buff_runs = Arc::new(AtomicUsize::new(0));
        let b1 = Arc::clone(&buff_runs);
        world.spawn_buff(
            id,
            1,
            move |ctx| async move {
                ctx.sleep(WaitTimeout::WorldMs(50))
                    .await
                    .expect("sleep failed");
                b1.fetch_add(
                    1,
                    Ordering::SeqCst,
                );
            },
        );
        // Refreshing same buff key must cancel the first buff.
        let b2 = Arc::clone(&buff_runs);
        world.spawn_buff(
            id,
            1,
            move |ctx| async move {
                ctx.sleep(WaitTimeout::WorldMs(50))
                    .await
                    .expect("sleep failed");
                b2.fetch_add(
                    10,
                    Ordering::SeqCst,
                );
            },
        );
        drain(
            &mut world, &mut slab,
        );

        // Pause world clock -> Ui advances, World stays 0.
        world.clocks.set_world_speed(0.0);
        world.clocks.advance(100);
        let mut world = slab.poll_ready(world);
        assert_eq!(
            world.clocks.ui_time_ms(),
            100
        );
        assert_eq!(
            world.clocks.world_time_ms(),
            0
        );
        assert_eq!(
            buff_runs.load(Ordering::SeqCst),
            0
        );

        // Resume world clock -> only refreshed buff completes (+10, not +1).
        world.clocks.set_world_speed(1.0);
        let dt = world.clocks.advance(60);
        world.physics.step(dt);
        let mut world = slab.poll_ready(world);
        assert_eq!(
            buff_runs.load(Ordering::SeqCst),
            10
        );

        // Despawning actor expires ActorId (returns None).
        assert!(world.despawn_actor(id));
        assert!(
            world
                .read_actor(
                    id,
                    |a| a.hp
                )
                .is_none()
        );
        let _ = InputBatch::default();
    }

    #[test]
    fn action_channel_and_state_signal_respect_wait_timeout() {
        let (tx, rx) = sync::action_channel::<u32>(4);
        let sig = StateSignal::new(0_u32);
        let mut world = Box::new(
            WorldState::<TestApp>::new(
                tx.clone(),
                rx.clone(),
            ),
        );
        let mut slab = TaskSlab::new();
        let results = Arc::new(std::sync::Mutex::new(Vec::new()));
        let res_clone = Arc::clone(&results);

        let rx_clone = rx.clone();
        let sig_clone = sig.clone();
        world.spawn_global_script(
            move |_ctx| async move {
                let r1 = rx_clone.next(WaitTimeout::UiMs(20)).await;
                let r2 = sig_clone
                    .wait_until(
                        WaitTimeout::UiMs(50),
                        |v| (*v >= 42).then_some(*v),
                    )
                    .await;
                res_clone.lock().expect("mutex poisoned").push(
                    (
                        r1.is_err(),
                        r2,
                    ),
                );
            },
        );
        drain(
            &mut world, &mut slab,
        );
        let mut world = slab.poll_ready(world);

        world.clocks.advance(25);
        let world = slab.poll_ready(world);
        sig.set(42);
        let _world = slab.poll_ready(world);

        let guard = results.lock().expect("mutex poisoned");
        assert_eq!(
            guard.len(),
            1
        );
        assert_eq!(
            guard[0],
            (
                true,
                Ok(42)
            )
        );
    }

    #[test]
    fn compute_worker_pool_and_rapier_spatial_query() {
        let (tx, rx) = sync::action_channel(4);
        let mut world = Box::new(
            WorldState::<TestApp>::new(
                tx, rx,
            ),
        );
        let mut slab = TaskSlab::new();

        let id = world.spawn_actor(
            ActorSpawnDesc {
                position: glam::Vec3::new(
                    0.0, 1.0, 0.0,
                ),
                rotation: glam::Quat::IDENTITY,
                half_extents: glam::Vec3::splat(0.5),
                color: [0.0, 1.0, 0.0],
                effect: None,
                body_kind: BodyKind::Fixed,
                mass: 1.0,
                data: TestActor { hp: 100 },
            },
        );
        world.physics.step(0.016);

        let hit = world.physics.raycast(
            glam::Vec3::new(
                0.0, 5.0, 0.0,
            ),
            glam::Vec3::NEG_Y,
            10.0,
            None,
        );
        assert_eq!(
            hit.and_then(|h| h.actor),
            Some(id)
        );

        let octree_hits = world.physics.query_octree(
            Aabb3::from_center_half_extents(
                glam::Vec3::new(
                    0.0, 1.0, 0.0,
                ),
                glam::Vec3::splat(2.0),
            ),
        );
        assert!(octree_hits.contains(&id));

        let done = Arc::new(AtomicUsize::new(0));
        let done_clone = Arc::clone(&done);
        world.spawn_global_script(
            move |ctx| async move {
                let task = ctx
                    .spawn_compute(
                        21_usize,
                        |v| async move { v * 2 },
                    )
                    .expect("worker pool available");
                let ans = task.await;
                done_clone.store(
                    ans,
                    Ordering::SeqCst,
                );
            },
        );
        drain(
            &mut world, &mut slab,
        );
        for _ in 0..50 {
            world = slab.poll_ready(world);
            if done.load(Ordering::SeqCst) == 42 {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        assert_eq!(
            done.load(Ordering::SeqCst),
            42
        );
    }

    #[test]
    fn despawn_keeps_octree_and_render_meshes_in_sync() {
        let (tx, rx) = sync::action_channel(4);
        let mut world = Box::new(
            WorldState::<TestApp>::new(
                tx, rx,
            ),
        );
        let mut slab = TaskSlab::new();

        let mut ids = Vec::new();
        for i in 0..7 {
            ids.push(
                world.spawn_actor(
                    ActorSpawnDesc {
                        position: glam::Vec3::new(
                            i as f32 * 2.0,
                            1.0,
                            0.0,
                        ),
                        rotation: glam::Quat::IDENTITY,
                        half_extents: glam::Vec3::splat(0.5),
                        color: [0.5, 0.5, 0.5],
                        effect: None,
                        body_kind: BodyKind::Fixed,
                        mass: 1.0,
                        data: TestActor { hp: 10 },
                    },
                ),
            );
        }

        world.physics.step(0.016);
        let target = ids[5];
        world.spawn_global_script(
            move |ctx| async move {
                assert!(ctx.despawn_actor(target));
            },
        );
        drain(
            &mut world, &mut slab,
        );
        let world = slab.poll_ready(world);

        let meshes = world.collect_render_meshes(&CameraSnapshot::default());
        assert_eq!(
            meshes.len(),
            6
        );
        assert!(meshes.iter().all(|m| m.actor != target));
    }

    #[test]
    fn physics_set_rotation_updates_soa_and_actor_rotation() {
        let (tx, rx) = sync::action_channel(4);
        let mut world = Box::new(
            WorldState::<TestApp>::new(
                tx, rx,
            ),
        );
        let id = world.spawn_actor(
            ActorSpawnDesc {
                position: glam::Vec3::new(
                    0.0, 1.0, 0.0,
                ),
                rotation: glam::Quat::IDENTITY,
                half_extents: glam::Vec3::splat(0.5),
                color: [0.2, 0.5, 0.9],
                effect: None,
                body_kind: BodyKind::Dynamic,
                mass: 1.0,
                data: TestActor { hp: 100 },
            },
        );
        let expected_rot = glam::Quat::from_rotation_y(-std::f32::consts::FRAC_PI_2);
        world.pending_physics_cmds.push(
            PhysicsCommand::SetRotation {
                actor: id,
                rotation: expected_rot,
            },
        );
        world
            .physics
            .flush_commands(&mut world.pending_physics_cmds);
        world.physics.step(0.016);

        let actual_rot = world.read_view().actor_rotation(id);
        assert!(actual_rot.is_some_and(|r| r.dot(expected_rot).abs() > 0.999));
    }

    #[test]
    fn mesh_effect_attachment_updates_render_snapshot_and_checkpoints() {
        struct DummyEffect;
        impl RenderEffect for DummyEffect {
            fn shader_spec(&self) -> EffectShaderSpec {
                EffectShaderSpec {
                    label: "Dummy Effect",
                    spirv_bytes: include_bytes!(env!("CORE_SHADER_SPV")),
                    vertex_entry: "vs_main",
                    fragment_entry: "fs_main",
                    blend: EffectBlendMode::AlphaBlend,
                }
            }
        }

        let mut registry = EffectRegistry::new();
        let fx_id = registry.register(DummyEffect);
        assert_eq!(
            registry.into_effects().len(),
            1
        );

        let (tx, rx) = sync::action_channel(4);
        let mut world = Box::new(
            WorldState::<TestApp>::new(
                tx, rx,
            ),
        );
        let id = world.spawn_actor(
            ActorSpawnDesc {
                position: glam::Vec3::new(
                    0.0, 1.0, 0.0,
                ),
                rotation: glam::Quat::IDENTITY,
                half_extents: glam::Vec3::splat(0.5),
                color: [0.2, 0.5, 0.9],
                effect: None,
                body_kind: BodyKind::Fixed,
                mass: 1.0,
                data: TestActor { hp: 100 },
            },
        );

        let attachment = EffectAttachment {
            effect: fx_id,
            params: [[1.0, 0.5, 0.2, 0.9], [2.0, 0.0, 0.0, 0.0]],
        };
        world.pending_physics_cmds.push(
            PhysicsCommand::SetMeshEffect {
                actor: id,
                effect: Some(attachment),
            },
        );
        world
            .physics
            .flush_commands(&mut world.pending_physics_cmds);

        let meshes = world.collect_render_meshes(&CameraSnapshot::default());
        assert_eq!(
            meshes.len(),
            1
        );
        assert_eq!(
            meshes[0].effect,
            Some(attachment)
        );

        let checkpoints = world.collect_actor_checkpoints();
        assert_eq!(
            checkpoints.len(),
            1
        );
        assert_eq!(
            checkpoints[0].effect,
            Some(attachment)
        );
    }

    #[test]
    fn core_shader_compiles_on_vulkan_1_3() {
        let spec = EffectShaderSpec {
            label: "Core Scene Shader",
            spirv_bytes: include_bytes!(env!("CORE_SHADER_SPV")),
            vertex_entry: "vs_main",
            fragment_entry: "fs_main",
            blend: EffectBlendMode::Opaque,
        };
        validate_effect_shader_vulkan(spec)
            .expect("core_shader SPIR-V must compile into a Vulkan 1.3 pipeline");
    }
}
