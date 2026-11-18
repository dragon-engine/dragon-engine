#![no_std]

/// Descriptor set index for engine-managed scene uniforms.
pub const SCENE_BIND_GROUP_INDEX: u32 = 0;

/// Descriptor set index reserved for effect-specific bindings.
pub const EFFECT_BIND_GROUP_INDEX: u32 = 1;

/// Scalar-backed 16-byte vector matching Vulkan `std140` uniform buffer rules without array stride issues.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[cfg_attr(feature = "bytemuck", derive(bytemuck::Pod, bytemuck::Zeroable))]
pub struct Vec4f {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

impl Vec4f {
    #[must_use]
    pub const fn new(x: f32, y: f32, z: f32, w: f32) -> Self {
        Self { x, y, z, w }
    }

    #[must_use]
    pub const fn from_array(a: [f32; 4]) -> Self {
        Self {
            x: a[0],
            y: a[1],
            z: a[2],
            w: a[3],
        }
    }

    #[must_use]
    pub const fn to_array(self) -> [f32; 4] {
        [self.x, self.y, self.z, self.w]
    }
}

/// Scalar-backed 4x4 column-major matrix matching Vulkan `std140` uniform buffer rules.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[cfg_attr(feature = "bytemuck", derive(bytemuck::Pod, bytemuck::Zeroable))]
pub struct Mat4f {
    pub col0: Vec4f,
    pub col1: Vec4f,
    pub col2: Vec4f,
    pub col3: Vec4f,
}

impl Mat4f {
    #[must_use]
    pub const fn from_cols_array_2d(m: [[f32; 4]; 4]) -> Self {
        Self {
            col0: Vec4f::from_array(m[0]),
            col1: Vec4f::from_array(m[1]),
            col2: Vec4f::from_array(m[2]),
            col3: Vec4f::from_array(m[3]),
        }
    }

    #[must_use]
    pub const fn to_cols_array_2d(self) -> [[f32; 4]; 4] {
        [
            self.col0.to_array(),
            self.col1.to_array(),
            self.col2.to_array(),
            self.col3.to_array(),
        ]
    }
}

/// Shared per-frame uniform payload bound at `@group(0) @binding(0)` for scene and effect shaders.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[cfg_attr(feature = "bytemuck", derive(bytemuck::Pod, bytemuck::Zeroable))]
pub struct SceneUniforms {
    pub view_proj: Mat4f,
    /// `(camera_eye.x, camera_eye.y, camera_eye.z, world_time_secs)`
    pub camera_eye_time: Vec4f,
    /// `(screen_width, screen_height, ui_time_secs, 0.0)`
    pub screen_and_ui_time: Vec4f,
}

/// Shared vertex layout emitted by `engine` for base scene passes and overlay effect passes.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[cfg_attr(feature = "bytemuck", derive(bytemuck::Pod, bytemuck::Zeroable))]
pub struct SceneVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
    pub color: [f32; 3],
}
