use crate::effect::EffectAttachment;
use crate::physics::ActorId;
use crate::runner::RenderSchedule;

/// Immutable 3D box primitive published inside `RenderSnapshot`.
#[derive(Debug, Clone)]
pub struct RenderMesh {
    pub actor: ActorId,
    pub position: glam::Vec3,
    pub rotation: glam::Quat,
    pub half_extents: glam::Vec3,
    pub color: [f32; 3],
    pub effect: Option<EffectAttachment>,
}

/// Camera parameters published in `RenderSnapshot`.
#[derive(Debug, Clone, Copy)]
pub struct CameraSnapshot {
    pub eye: glam::Vec3,
    pub target: glam::Vec3,
    pub up: glam::Vec3,
    pub fov_y_radians: f32,
}

/// Game-agnostic 3D box gizmo primitive published in `RenderSnapshot`.
#[derive(Debug, Clone, Copy)]
pub struct GizmoBox {
    pub position: glam::Vec3,
    pub rotation: glam::Quat,
    pub half_extents: glam::Vec3,
    pub color: [f32; 3],
}

impl Default for CameraSnapshot {
    fn default() -> Self {
        Self {
            eye: glam::Vec3::new(
                0.0, 14.0, 16.0,
            ),
            target: glam::Vec3::ZERO,
            up: glam::Vec3::Y,
            fov_y_radians: 45.0_f32.to_radians(),
        }
    }
}

/// Completed frame snapshot read by `Renderer` and `UiOverlay`.
/// `EngineRunner` owns two instances so `game` never reads incomplete frames.
#[derive(Debug, Clone)]
pub struct RenderSnapshot<G> {
    pub frame_index: u64,
    pub world_time_ms: u64,
    pub ui_time_ms: u64,
    pub world_speed: f32,
    pub schedule: RenderSchedule,
    pub camera: CameraSnapshot,
    pub meshes: Vec<RenderMesh>,
    pub gizmos: Vec<GizmoBox>,
    pub cursor_grab: bool,
    pub game: G,
}

impl<G: Default> Default for RenderSnapshot<G> {
    fn default() -> Self {
        Self {
            frame_index: 0,
            world_time_ms: 0,
            ui_time_ms: 0,
            world_speed: 0.0,
            schedule: RenderSchedule::Sequential,
            camera: CameraSnapshot::default(),
            meshes: Vec::new(),
            gizmos: Vec::new(),
            cursor_grab: false,
            game: G::default(),
        }
    }
}
