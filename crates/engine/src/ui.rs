use crate::snapshot::RenderSnapshot;
use crate::sync::ActionSender;

pub(crate) trait UiCompositor {
    fn composite_rgba8(&mut self, width: u32, height: u32, rgba8_pixels: &[u8]);
}

/// Graphics target handed to `UiOverlay::render` after the 3D scene pass finishes.
/// Keeps Vulkan handles private inside `crates/engine`.
pub struct UiRenderTarget<'a> {
    pub width: u32,
    pub height: u32,
    pub scale_factor: f32,
    compositor: &'a mut dyn UiCompositor,
}

impl<'a> UiRenderTarget<'a> {
    pub(crate) fn new(
        width: u32,
        height: u32,
        scale_factor: f32,
        compositor: &'a mut dyn UiCompositor,
    ) -> Self {
        Self {
            width,
            height,
            scale_factor,
            compositor,
        }
    }

    /// Uploads premultiplied RGBA8 pixels and composites them over the active frame target.
    pub fn composite_rgba8(&mut self, rgba8_pixels: &[u8]) {
        self.compositor.composite_rgba8(
            self.width,
            self.height,
            rgba8_pixels,
        );
    }
}

/// UI abstraction decoupled from `GameApp`.
/// Implementations may use `iced`, `egui`, custom `taffy`, or a hybrid approach.
pub trait UiOverlay<G, A>: 'static {
    /// Inspects raw `winit` events first. Returning `true` blocks the event from `GameApp::on_input`.
    fn handle_event(
        &mut self,
        event: &winit::event::WindowEvent,
        window: &winit::window::Window,
    ) -> bool;

    /// Synchronizes UI state from the latest completed `RenderSnapshot` and pushes UI actions
    /// into the shared FIFO `ActionSender<A>`.
    fn sync(&mut self, snapshot: &RenderSnapshot<G>, actions: &ActionSender<A>);

    /// Renders UI elements onto `UiRenderTarget` after the 3D scene pass.
    fn render(&mut self, target: &mut UiRenderTarget<'_>, ui_time_ms: u64);
}

/// Stateless helper that composites premultiplied RGBA8 UI pixel buffers onto `UiRenderTarget`.
#[derive(Debug, Default)]
pub struct UiBlitter;

impl UiBlitter {
    /// Creates a `UiBlitter` bound to `engine`'s internal Vulkan UI compositor.
    #[must_use]
    pub fn new(_target: &UiRenderTarget<'_>) -> Self {
        Self
    }

    /// Uploads premultiplied RGBA8 pixels and composites them over `target`.
    pub fn composite_rgba8(&mut self, target: &mut UiRenderTarget<'_>, rgba8_pixels: &[u8]) {
        target.composite_rgba8(rgba8_pixels);
    }
}
