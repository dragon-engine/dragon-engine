use crate::snapshot::RenderSnapshot;
use crate::sync::ActionSender;

/// Graphics target handed to `UiOverlay::render` after the 3D scene pass finishes.
pub struct UiRenderTarget<'a> {
    pub device: &'a wgpu::Device,
    pub queue: &'a wgpu::Queue,
    pub view: &'a wgpu::TextureView,
    pub encoder: &'a mut wgpu::CommandEncoder,
    pub format: wgpu::TextureFormat,
    pub width: u32,
    pub height: u32,
    pub scale_factor: f32,
}

/// UI abstraction decoupled from `GameApp`.
/// Implementations may use `iced`, `egui`, custom `taffy` + `wgpu`, or a hybrid approach.
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

    /// Renders UI elements onto the `wgpu` surface target after the 3D scene pass.
    fn render(&mut self, target: &mut UiRenderTarget<'_>, ui_time_ms: u64);
}

/// Composites premultiplied RGBA8 UI pixel buffers onto `UiRenderTarget` using `rust-gpu` SPIR-V shaders.
pub struct UiBlitter {
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    texture: wgpu::Texture,
    bind_group: wgpu::BindGroup,
    width: u32,
    height: u32,
}

impl UiBlitter {
    /// Creates a new `UiBlitter` targeting `target`'s surface format and dimensions.
    pub fn new(target: &UiRenderTarget<'_>) -> Self {
        let shader = target.device.create_shader_module(
            wgpu::ShaderModuleDescriptor {
                label: Some("Rust-GPU UI Shader"),
                source: wgpu::ShaderSource::SpirV(
                    wgpu::util::make_spirv_raw(include_bytes!(env!("CORE_SHADER_SPV"))),
                ),
            },
        );

        let bind_group_layout = target.device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("UI Blit BindGroupLayout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            },
        );

        let pipeline_layout = target.device.create_pipeline_layout(
            &wgpu::PipelineLayoutDescriptor {
                label: Some("UI Blit PipelineLayout"),
                bind_group_layouts: &[Some(&bind_group_layout)],
                immediate_size: 0,
            },
        );

        let pipeline = target.device.create_render_pipeline(
            &wgpu::RenderPipelineDescriptor {
                label: Some("Rust-GPU UI Blit Pipeline"),
                layout: Some(&pipeline_layout),
                cache: None,
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("ui_vs_main"),
                    buffers: &[],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(
                    wgpu::FragmentState {
                        module: &shader,
                        entry_point: Some("ui_fs_main"),
                        targets: &[Some(
                            wgpu::ColorTargetState {
                                format: target.format,
                                blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                                write_mask: wgpu::ColorWrites::ALL,
                            },
                        )],
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                    },
                ),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
            },
        );

        let sampler = target.device.create_sampler(
            &wgpu::SamplerDescriptor {
                label: Some("UI Sampler"),
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            },
        );

        let width = target.width.max(1);
        let height = target.height.max(1);
        let (texture, bind_group) = Self::create_texture(
            target.device,
            &bind_group_layout,
            &sampler,
            width,
            height,
        );

        Self {
            pipeline,
            bind_group_layout,
            sampler,
            texture,
            bind_group,
            width,
            height,
        }
    }

    fn create_texture(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        sampler: &wgpu::Sampler,
        width: u32,
        height: u32,
    ) -> (
        wgpu::Texture,
        wgpu::BindGroup,
    ) {
        let texture = device.create_texture(
            &wgpu::TextureDescriptor {
                label: Some("UI Overlay Texture"),
                size: wgpu::Extent3d {
                    width: width.max(1),
                    height: height.max(1),
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            },
        );
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let bind_group = device.create_bind_group(
            &wgpu::BindGroupDescriptor {
                label: Some("UI Blit BindGroup"),
                layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(sampler),
                    },
                ],
            },
        );
        (
            texture, bind_group,
        )
    }

    /// Uploads premultiplied RGBA8 pixels and composites them over `target.view`.
    pub fn composite_rgba8(&mut self, target: &mut UiRenderTarget<'_>, rgba8_pixels: &[u8]) {
        let width = target.width.max(1);
        let height = target.height.max(1);
        if self.width != width || self.height != height {
            let (texture, bind_group) = Self::create_texture(
                target.device,
                &self.bind_group_layout,
                &self.sampler,
                width,
                height,
            );
            self.texture = texture;
            self.bind_group = bind_group;
            self.width = width;
            self.height = height;
        }

        target.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            rgba8_pixels,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * width),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );

        let mut pass = target.encoder.begin_render_pass(
            &wgpu::RenderPassDescriptor {
                label: Some("Rust-GPU UI Composite Pass"),
                color_attachments: &[Some(
                    wgpu::RenderPassColorAttachment {
                        view: target.view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    },
                )],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
                multiview_mask: None,
            },
        );

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(
            0,
            &self.bind_group,
            &[],
        );
        pass.draw(
            0..3,
            0..1,
        );
    }
}
