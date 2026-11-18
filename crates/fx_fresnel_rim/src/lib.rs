use std::num::NonZeroU64;

use engine::{EffectAttachment, EffectDrawItem, EffectId, EffectInitContext, RenderEffect, wgpu};

const PARAM_BYTES: usize = size_of::<[[f32; 4]; 2]>();

/// Typed configuration helper that packs Fresnel rim parameters into `EffectAttachment`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FresnelRimParams {
    pub color: [f32; 3],
    pub intensity: f32,
    pub power: f32,
    pub pulse_speed: f32,
    pub edge_boost: f32,
}

impl FresnelRimParams {
    #[must_use]
    pub fn new(color: [f32; 3], intensity: f32, power: f32) -> Self {
        Self {
            color,
            intensity,
            power,
            pulse_speed: 0.0,
            edge_boost: 0.55,
        }
    }

    #[must_use]
    pub fn with_pulse(mut self, pulse_speed: f32) -> Self {
        self.pulse_speed = pulse_speed;
        self
    }

    #[must_use]
    pub fn with_edge_boost(mut self, edge_boost: f32) -> Self {
        self.edge_boost = edge_boost;
        self
    }

    #[must_use]
    pub fn attach(self, effect: EffectId) -> EffectAttachment {
        EffectAttachment {
            effect,
            params: [
                [self.color[0], self.color[1], self.color[2], self.intensity],
                [self.power, self.pulse_speed, self.edge_boost, 0.0],
            ],
        }
    }
}

struct RimGpuState {
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    uniform_stride: usize,
    capacity_items: usize,
}

/// Pluggable Fresnel Rim overlay effect implemented with `rust-gpu`.
#[derive(Default)]
pub struct FresnelRimEffect {
    gpu: Option<RimGpuState>,
}

impl FresnelRimEffect {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    fn create_uniform_buffer_and_bind_group(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        stride: usize,
        capacity_items: usize,
    ) -> (
        wgpu::Buffer,
        wgpu::BindGroup,
    ) {
        let buffer = device.create_buffer(
            &wgpu::BufferDescriptor {
                label: Some("Fresnel Rim Uniform Buffer"),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                size: (stride * capacity_items.max(1)) as u64,
                mapped_at_creation: false,
            },
        );
        let binding_size = NonZeroU64::new(PARAM_BYTES as u64)
            .unwrap_or_else(|| unreachable!("PARAM_BYTES is non-zero"));
        let bind_group = device.create_bind_group(
            &wgpu::BindGroupDescriptor {
                label: Some("Fresnel Rim BindGroup"),
                layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(
                        wgpu::BufferBinding {
                            buffer: &buffer,
                            offset: 0,
                            size: Some(binding_size),
                        },
                    ),
                }],
            },
        );
        (
            buffer, bind_group,
        )
    }
}

impl RenderEffect for FresnelRimEffect {
    fn init(&mut self, ctx: &EffectInitContext<'_>) {
        let shader = ctx.device.create_shader_module(
            wgpu::ShaderModuleDescriptor {
                label: Some("Fresnel Rim Shader"),
                source: wgpu::ShaderSource::SpirV(
                    wgpu::util::make_spirv_raw(include_bytes!(env!("FRESNEL_RIM_SHADER_SPV"))),
                ),
            },
        );

        let min_align = ctx.device.limits().min_uniform_buffer_offset_alignment as usize;
        let uniform_stride = PARAM_BYTES.next_multiple_of(min_align.max(32));
        let binding_size = NonZeroU64::new(PARAM_BYTES as u64)
            .unwrap_or_else(|| unreachable!("PARAM_BYTES is non-zero"));

        let bind_group_layout = ctx.device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("fresnel_rim_bind_group_layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: Some(binding_size),
                    },
                    count: None,
                }],
            },
        );

        let pipeline_layout = ctx.device.create_pipeline_layout(
            &wgpu::PipelineLayoutDescriptor {
                label: Some("Fresnel Rim Pipeline Layout"),
                bind_group_layouts: &[Some(ctx.scene_bind_group_layout), Some(&bind_group_layout)],
                immediate_size: 0,
            },
        );

        let pipeline = ctx.device.create_render_pipeline(
            &wgpu::RenderPipelineDescriptor {
                cache: None,
                label: Some("Fresnel Rim Render Pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[Some(ctx.vertex_layout.clone())],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(
                    wgpu::FragmentState {
                        module: &shader,
                        entry_point: Some("fs_main"),
                        targets: &[Some(
                            wgpu::ColorTargetState {
                                format: ctx.target_format,
                                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                                write_mask: wgpu::ColorWrites::ALL,
                            },
                        )],
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                    },
                ),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: Some(wgpu::Face::Back),
                    ..Default::default()
                },
                depth_stencil: Some(
                    wgpu::DepthStencilState {
                        format: ctx.depth_format,
                        // Overlay pass shares the base depth buffer so occluded faces do not bleed through.
                        depth_write_enabled: Some(false),
                        depth_compare: Some(wgpu::CompareFunction::LessEqual),
                        stencil: wgpu::StencilState::default(),
                        bias: wgpu::DepthBiasState::default(),
                    },
                ),
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
            },
        );

        let initial_capacity = 16;
        let (uniform_buffer, bind_group) = Self::create_uniform_buffer_and_bind_group(
            ctx.device,
            &bind_group_layout,
            uniform_stride,
            initial_capacity,
        );

        self.gpu = Some(
            RimGpuState {
                pipeline,
                bind_group_layout,
                uniform_buffer,
                bind_group,
                uniform_stride,
                capacity_items: initial_capacity,
            },
        );
    }

    fn prepare(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, items: &[EffectDrawItem]) {
        let Some(gpu) = self.gpu.as_mut() else {
            return;
        };
        if items.is_empty() {
            return;
        }
        if items.len() > gpu.capacity_items {
            let new_cap = items.len().next_power_of_two();
            let (buffer, bind_group) = Self::create_uniform_buffer_and_bind_group(
                device,
                &gpu.bind_group_layout,
                gpu.uniform_stride,
                new_cap,
            );
            gpu.uniform_buffer = buffer;
            gpu.bind_group = bind_group;
            gpu.capacity_items = new_cap;
        }

        let mut staging = vec![0_u8; items.len() * gpu.uniform_stride];
        for (i, item) in items.iter().enumerate() {
            let offset = i * gpu.uniform_stride;
            staging[offset..offset + PARAM_BYTES].copy_from_slice(bytemuck::bytes_of(&item.params));
        }
        queue.write_buffer(
            &gpu.uniform_buffer,
            0,
            &staging,
        );
    }

    fn draw<'pass>(&'pass self, pass: &mut wgpu::RenderPass<'pass>, items: &[EffectDrawItem]) {
        let Some(gpu) = self.gpu.as_ref() else {
            return;
        };
        if items.is_empty() {
            return;
        }
        pass.set_pipeline(&gpu.pipeline);
        for (i, item) in items.iter().enumerate() {
            let dynamic_offset = u32::try_from(i * gpu.uniform_stride)
                .unwrap_or_else(|err| panic!("Uniform offset overflow: {err}"));
            pass.set_bind_group(
                engine::shader_abi::EFFECT_BIND_GROUP_INDEX,
                &gpu.bind_group,
                &[dynamic_offset],
            );
            pass.draw(
                item.vertex_range.clone(),
                0..1,
            );
        }
    }
}

#[cfg(test)]
#[allow(clippy::disallowed_methods)]
mod tests {
    use super::*;

    #[test]
    fn fresnel_rim_params_pack_into_effect_attachment() {
        let mut reg = engine::EffectRegistry::new();
        let fx_id = reg.register(FresnelRimEffect::new());
        let attachment = FresnelRimParams::new(
            [0.2, 0.8, 1.0],
            0.9,
            2.5,
        )
        .with_pulse(4.0)
        .with_edge_boost(0.35)
        .attach(fx_id);

        assert_eq!(
            attachment.effect,
            fx_id
        );
        assert_eq!(
            attachment.params[0],
            [0.2, 0.8, 1.0, 0.9]
        );
        assert_eq!(
            attachment.params[1],
            [2.5, 4.0, 0.35, 0.0]
        );
    }

    #[test]
    fn fresnel_rim_shader_and_pipeline_validate_in_wgpu() {
        let instance = wgpu::Instance::default();
        let Ok(adapter) =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
        else {
            return;
        };
        let Ok((device, _queue)) = pollster::block_on(
            adapter.request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Test Device"),
                    memory_hints: wgpu::MemoryHints::Performance,
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::downlevel_webgl2_defaults()
                        .using_resolution(adapter.limits()),
                    experimental_features: wgpu::ExperimentalFeatures::disabled(),
                    trace: wgpu::Trace::Off,
                },
            ),
        ) else {
            return;
        };

        let scene_layout = device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("test_scene_layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            },
        );
        const ATTRS: [wgpu::VertexAttribute; 4] = wgpu::vertex_attr_array![
            0 => Float32x3,
            1 => Float32x3,
            2 => Float32x2,
            3 => Float32x3
        ];
        let ctx = EffectInitContext {
            device: &device,
            scene_bind_group_layout: &scene_layout,
            vertex_layout: wgpu::VertexBufferLayout {
                array_stride: size_of::<engine::shader_abi::SceneVertex>() as wgpu::BufferAddress,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &ATTRS,
            },
            target_format: wgpu::TextureFormat::Bgra8UnormSrgb,
            depth_format: wgpu::TextureFormat::Depth32Float,
        };
        let mut effect = FresnelRimEffect::new();
        effect.init(&ctx);
    }
}
