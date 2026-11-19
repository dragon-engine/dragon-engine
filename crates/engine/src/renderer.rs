use std::sync::Arc;

use winit::dpi::PhysicalSize;
use winit::window::Window;

use crate::effect::EffectRegistry;
use crate::gpu::GpuError;
use crate::gpu::renderer::VulkanRenderer;
use crate::snapshot::RenderSnapshot;
use crate::ui::UiOverlay;

pub(crate) struct Renderer {
    inner: VulkanRenderer,
}

impl Renderer {
    pub(crate) fn new(
        window: Arc<Window>,
        size: PhysicalSize<u32>,
        effects: EffectRegistry,
    ) -> Result<Self, GpuError> {
        let inner = VulkanRenderer::new(
            window, size, effects,
        )?;
        Ok(Self { inner })
    }

    pub(crate) fn resize(&mut self, size: PhysicalSize<u32>) {
        self.inner.resize(size);
    }

    pub(crate) fn draw<G, A, U: UiOverlay<G, A>>(
        &mut self,
        snapshot: &RenderSnapshot<G>,
        ui: &mut U,
    ) {
        self.inner.draw(
            snapshot, ui,
        );
    }
}
