use std::sync::Arc;

use wgpu::{self, ExperimentalFeatures, Trace};
use winit::dpi::PhysicalSize;
use winit::window::Window;

use crate::effect::EffectRegistry;
use crate::render_pipeline::SceneRenderPipeline;
use crate::snapshot::RenderSnapshot;
use crate::ui::{UiOverlay, UiRenderTarget};

pub(crate) struct Renderer<'a> {
    instance: wgpu::Instance,
    window: Arc<Window>,
    queue: wgpu::Queue,
    device: wgpu::Device,
    surface_config: wgpu::SurfaceConfiguration,
    surface: wgpu::Surface<'a>,
    scene_pipeline: SceneRenderPipeline,
}

impl Renderer<'_> {
    pub(crate) fn new<'window>(
        window: Arc<Window>,
        size: PhysicalSize<u32>,
        effects: EffectRegistry,
    ) -> Renderer<'window> {
        let instance = wgpu::Instance::default();
        let surface = instance
            .create_surface(window.clone())
            .unwrap_or_else(|e| panic!("Failed to create wgpu surface: {e}"));

        let adapter = pollster::block_on(
            instance.request_adapter(
                &wgpu::RequestAdapterOptions {
                    apply_limit_buckets: false,
                    power_preference: wgpu::PowerPreference::default(),
                    force_fallback_adapter: false,
                    compatible_surface: Some(&surface),
                },
            ),
        )
        .unwrap_or_else(|e| panic!("Failed to find suitable GPU adapter: {e}"));

        // TODO(wgpu): `wgpu 30.0.1` does not implement incremental BLAS/TLAS refit for
        // `AccelerationStructureUpdateMode::PreferUpdate` and `AccelerationStructureFlags::ALLOW_UPDATE`;
        // `wgpu-core` always performs a full rebuild. Set `ALLOW_UPDATE | PREFER_FAST_BUILD` and
        // `PreferUpdate` on dynamic BLASes, and cap dynamic ray-traced characters until upstream refit lands:
        // https://github.com/gfx-rs/wgpu/issues/6762
        //
        // TODO(wgpu): `wgpu::Blas` does not expose allocated or compacted byte sizes (`size_info` is
        // private inside `wgpu-core`). Track `BLAS_POOL_BUDGET_MB` with a CPU triangle-count estimator
        // inside a private `BlasPool` module until `wgpu` exposes BLAS memory queries:
        // https://github.com/gfx-rs/wgpu/issues/6762
        // https://github.com/gfx-rs/wgpu/pull/7285
        let (device, queue) = pollster::block_on(
            adapter.request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Engine Device"),
                    memory_hints: wgpu::MemoryHints::Performance,
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::downlevel_webgl2_defaults()
                        .using_resolution(adapter.limits()),
                    experimental_features: ExperimentalFeatures::disabled(),
                    trace: Trace::Off,
                },
            ),
        )
        .unwrap_or_else(|e| panic!("Failed to create GPU device: {e}"));

        let surface_config = surface
            .get_default_config(
                &adapter,
                size.width.max(1),
                size.height.max(1),
            )
            .expect("surface incompatible with adapter");

        surface.configure(
            &device,
            &surface_config,
        );
        let scene_pipeline = SceneRenderPipeline::new(
            &device,
            &surface_config,
            effects,
        );

        Renderer {
            instance,
            window,
            queue,
            device,
            surface_config,
            surface,
            scene_pipeline,
        }
    }

    fn configure_surface(&self) {
        self.surface.configure(
            &self.device,
            &self.surface_config,
        );
    }

    pub(crate) fn resize(&mut self, size: PhysicalSize<u32>) {
        if size.width == 0 || size.height == 0 {
            return;
        }
        self.surface_config.width = size.width;
        self.surface_config.height = size.height;
        self.configure_surface();
        self.scene_pipeline.resize(
            &self.device,
            size.width,
            size.height,
        );
    }

    pub(crate) fn draw<G, A, U: UiOverlay<G, A>>(
        &mut self,
        snapshot: &RenderSnapshot<G>,
        ui: &mut U,
    ) {
        let surface_texture = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(texture) => texture,
            wgpu::CurrentSurfaceTexture::Occluded | wgpu::CurrentSurfaceTexture::Timeout => return,
            wgpu::CurrentSurfaceTexture::Suboptimal(texture) => {
                drop(texture);
                self.configure_surface();
                return;
            }
            wgpu::CurrentSurfaceTexture::Outdated => {
                self.configure_surface();
                return;
            }
            wgpu::CurrentSurfaceTexture::Validation => {
                unreachable!("no validation error scope registered")
            }
            wgpu::CurrentSurfaceTexture::Lost => {
                self.surface = self
                    .instance
                    .create_surface(self.window.clone())
                    .unwrap_or_else(|e| panic!("Failed to recreate lost surface: {e}"));
                self.configure_surface();
                return;
            }
        };

        let target_format = self.surface_config.format.add_srgb_suffix();
        let texture_view = surface_texture.texture.create_view(
            &wgpu::TextureViewDescriptor {
                format: Some(target_format),
                ..Default::default()
            },
        );

        let mut encoder = self.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("Frame Encoder"),
            },
        );

        self.scene_pipeline.render(
            &self.device,
            &self.surface_config,
            &texture_view,
            &self.queue,
            &mut encoder,
            snapshot,
        );

        {
            let mut ui_target = UiRenderTarget {
                device: &self.device,
                queue: &self.queue,
                view: &texture_view,
                encoder: &mut encoder,
                format: target_format,
                width: self.surface_config.width,
                height: self.surface_config.height,
                scale_factor: self.window.scale_factor() as f32,
            };
            ui.render(
                &mut ui_target,
                snapshot.ui_time_ms,
            );
        }

        self.queue.submit(Some(encoder.finish()));
        self.window.pre_present_notify();
        self.queue.present(surface_texture);
    }
}
