use std::ops::Range;

use serde::{Deserialize, Serialize};

/// Opaque handle identifying a registered `RenderEffect` implementation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EffectId(pub(crate) u32);

impl EffectId {
    #[must_use]
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

/// Game-agnostic effect attachment stored per actor mesh in `engine` SoA storage.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct EffectAttachment {
    pub effect: EffectId,
    pub params: [[f32; 4]; 2],
}

/// Shared GPU resources provided by `engine` when initializing a `RenderEffect`.
pub struct EffectInitContext<'a> {
    pub device: &'a wgpu::Device,
    pub scene_bind_group_layout: &'a wgpu::BindGroupLayout,
    pub vertex_layout: wgpu::VertexBufferLayout<'static>,
    pub target_format: wgpu::TextureFormat,
    pub depth_format: wgpu::TextureFormat,
}

/// One mesh draw range with its attached opaque effect parameters.
#[derive(Debug, Clone)]
pub struct EffectDrawItem {
    pub vertex_range: Range<u32>,
    pub params: [[f32; 4]; 2],
}

/// Pluggable 3D overlay effect interface implemented by standard or external effect crates.
pub trait RenderEffect: Send + 'static {
    /// Initializes GPU shaders, bind group layouts, and render pipelines against `engine`'s ABI.
    fn init(&mut self, ctx: &EffectInitContext<'_>);

    /// Uploads per-frame uniform data for all meshes currently using this effect.
    fn prepare(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, items: &[EffectDrawItem]);

    /// Issues draw calls inside the active scene render pass (`@group(0)` and vertex buffer 0 are bound).
    fn draw<'pass>(&'pass self, pass: &mut wgpu::RenderPass<'pass>, items: &[EffectDrawItem]);
}

/// Registry passed to `EngineRunner::run_with_effects` to attach workspace or third-party effects.
#[derive(Default)]
pub struct EffectRegistry {
    effects: Vec<Box<dyn RenderEffect>>,
}

impl EffectRegistry {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register<E: RenderEffect>(&mut self, effect: E) -> EffectId {
        let id = u32::try_from(self.effects.len())
            .unwrap_or_else(|err| panic!("Too many registered render effects: {err}"));
        self.effects.push(Box::new(effect));
        EffectId(id)
    }

    pub(crate) fn into_effects(self) -> Vec<Box<dyn RenderEffect>> {
        self.effects
    }
}
