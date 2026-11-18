use shader_abi::{Mat4f, SCENE_BIND_GROUP_INDEX, SceneUniforms, SceneVertex, Vec4f};
use wgpu::util::DeviceExt;

use crate::effect::{EffectDrawItem, EffectInitContext, EffectRegistry, RenderEffect};
use crate::snapshot::RenderSnapshot;

const SCENE_VERTEX_ATTRIBUTES: [wgpu::VertexAttribute; 4] = wgpu::vertex_attr_array![
    0 => Float32x3,
    1 => Float32x3,
    2 => Float32x2,
    3 => Float32x3
];

fn scene_vertex_layout() -> wgpu::VertexBufferLayout<'static> {
    wgpu::VertexBufferLayout {
        array_stride: size_of::<SceneVertex>() as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &SCENE_VERTEX_ATTRIBUTES,
    }
}

const MAX_VERTICES: usize = 36 * 2048;

pub(crate) struct SceneRenderPipeline {
    render_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    scene_bind_group: wgpu::BindGroup,
    scene_uniform_buffer: wgpu::Buffer,
    depth_texture_view: wgpu::TextureView,
    effects: Vec<Box<dyn RenderEffect>>,
}

impl SceneRenderPipeline {
    pub(crate) fn new(
        device: &wgpu::Device,
        config: &wgpu::SurfaceConfiguration,
        effects: EffectRegistry,
    ) -> Self {
        let shader = device.create_shader_module(
            wgpu::ShaderModuleDescriptor {
                label: Some("Core Shader"),
                source: wgpu::ShaderSource::SpirV(
                    wgpu::util::make_spirv_raw(include_bytes!(env!("CORE_SHADER_SPV"))),
                ),
            },
        );

        let scene_bind_group_layout = device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
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
                label: Some("scene_bind_group_layout"),
            },
        );

        let scene_uniform_buffer = device.create_buffer(
            &wgpu::BufferDescriptor {
                label: Some("Scene Uniform Buffer"),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                size: size_of::<SceneUniforms>() as u64,
                mapped_at_creation: false,
            },
        );

        let scene_bind_group = device.create_bind_group(
            &wgpu::BindGroupDescriptor {
                layout: &scene_bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: scene_uniform_buffer.as_entire_binding(),
                }],
                label: Some("scene_bind_group"),
            },
        );

        let render_pipeline_layout = device.create_pipeline_layout(
            &wgpu::PipelineLayoutDescriptor {
                label: Some("Scene Pipeline Layout"),
                bind_group_layouts: &[Some(&scene_bind_group_layout)],
                immediate_size: 0,
            },
        );

        let target_format = config.format.add_srgb_suffix();
        let depth_format = wgpu::TextureFormat::Depth32Float;

        let render_pipeline = device.create_render_pipeline(
            &wgpu::RenderPipelineDescriptor {
                cache: None,
                label: Some("Scene Render Pipeline"),
                layout: Some(&render_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[Some(scene_vertex_layout())],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(
                    wgpu::FragmentState {
                        module: &shader,
                        entry_point: Some("fs_main"),
                        targets: &[Some(
                            wgpu::ColorTargetState {
                                format: target_format,
                                blend: Some(wgpu::BlendState::REPLACE),
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
                        format: depth_format,
                        depth_write_enabled: Some(true),
                        depth_compare: Some(wgpu::CompareFunction::Less),
                        stencil: wgpu::StencilState::default(),
                        bias: wgpu::DepthBiasState::default(),
                    },
                ),
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
            },
        );

        let vertex_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Scene Vertex Buffer"),
                contents: &vec![0u8; MAX_VERTICES * size_of::<SceneVertex>()],
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            },
        );

        let depth_texture_view = Self::create_depth_view(
            device,
            config.width,
            config.height,
        );

        let mut initialized_effects = effects.into_effects();
        let init_ctx = EffectInitContext {
            device,
            scene_bind_group_layout: &scene_bind_group_layout,
            vertex_layout: scene_vertex_layout(),
            target_format,
            depth_format,
        };
        for effect in &mut initialized_effects {
            effect.init(&init_ctx);
        }

        Self {
            render_pipeline,
            vertex_buffer,
            scene_bind_group,
            scene_uniform_buffer,
            depth_texture_view,
            effects: initialized_effects,
        }
    }

    pub(crate) fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        self.depth_texture_view = Self::create_depth_view(
            device,
            width.max(1),
            height.max(1),
        );
    }

    fn create_depth_view(device: &wgpu::Device, width: u32, height: u32) -> wgpu::TextureView {
        let texture = device.create_texture(
            &wgpu::TextureDescriptor {
                label: Some("Scene Depth Texture"),
                size: wgpu::Extent3d {
                    width: width.max(1),
                    height: height.max(1),
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Depth32Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            },
        );
        texture.create_view(&wgpu::TextureViewDescriptor::default())
    }

    pub(crate) fn render<G>(
        &mut self,
        device: &wgpu::Device,
        config: &wgpu::SurfaceConfiguration,
        view: &wgpu::TextureView,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        snapshot: &RenderSnapshot<G>,
    ) {
        let width = config.width.max(1) as f32;
        let height = config.height.max(1) as f32;
        let aspect = width / height;
        let proj = glam::camera::rh::proj::directx::perspective(
            snapshot.camera.fov_y_radians,
            aspect,
            0.1,
            250.0,
        );
        let view_mat = glam::camera::rh::view::look_at_mat4(
            snapshot.camera.eye,
            snapshot.camera.target,
            snapshot.camera.up,
        );
        let view_proj = (proj * view_mat).to_cols_array_2d();
        let eye = snapshot.camera.eye;
        let uniforms = SceneUniforms {
            view_proj: Mat4f::from_cols_array_2d(view_proj),
            camera_eye_time: Vec4f::new(
                eye.x,
                eye.y,
                eye.z,
                snapshot.world_time_ms as f32 * 0.001,
            ),
            screen_and_ui_time: Vec4f::new(
                width,
                height,
                snapshot.ui_time_ms as f32 * 0.001,
                0.0,
            ),
        };
        queue.write_buffer(
            &self.scene_uniform_buffer,
            0,
            bytemuck::bytes_of(&uniforms),
        );

        let mut vertices = Vec::with_capacity((snapshot.meshes.len() + snapshot.gizmos.len()) * 36);
        let mut effect_batches: Vec<Vec<EffectDrawItem>> = vec![Vec::new(); self.effects.len()];
        for mesh in &snapshot.meshes {
            if vertices.len() + 36 > MAX_VERTICES {
                break;
            }
            let start = vertices.len() as u32;
            push_box_vertices(
                &mut vertices,
                mesh.position,
                mesh.rotation,
                mesh.half_extents,
                mesh.color,
            );
            let end = vertices.len() as u32;
            if let Some(attachment) = mesh.effect
                && let Some(batch) = effect_batches.get_mut(attachment.effect.index())
            {
                batch.push(
                    EffectDrawItem {
                        vertex_range: start..end,
                        params: attachment.params,
                    },
                );
            }
        }
        for gizmo in &snapshot.gizmos {
            if vertices.len() + 36 > MAX_VERTICES {
                break;
            }
            push_box_vertices(
                &mut vertices,
                gizmo.position,
                gizmo.rotation,
                gizmo.half_extents,
                gizmo.color,
            );
        }

        if !vertices.is_empty() {
            queue.write_buffer(
                &self.vertex_buffer,
                0,
                bytemuck::cast_slice(&vertices),
            );
            for (effect, batch) in self.effects.iter_mut().zip(&effect_batches) {
                if !batch.is_empty() {
                    effect.prepare(
                        device, queue, batch,
                    );
                }
            }
        }

        let mut render_pass = encoder.begin_render_pass(
            &wgpu::RenderPassDescriptor {
                label: Some("3D Scene Pass"),
                color_attachments: &[Some(
                    wgpu::RenderPassColorAttachment {
                        view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(
                                wgpu::Color {
                                    r: 0.08,
                                    g: 0.11,
                                    b: 0.16,
                                    a: 1.0,
                                },
                            ),
                            store: wgpu::StoreOp::Store,
                        },
                    },
                )],
                depth_stencil_attachment: Some(
                    wgpu::RenderPassDepthStencilAttachment {
                        view: &self.depth_texture_view,
                        depth_ops: Some(
                            wgpu::Operations {
                                load: wgpu::LoadOp::Clear(1.0),
                                store: wgpu::StoreOp::Store,
                            },
                        ),
                        stencil_ops: None,
                    },
                ),
                occlusion_query_set: None,
                timestamp_writes: None,
                multiview_mask: None,
            },
        );

        if !vertices.is_empty() {
            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(
                SCENE_BIND_GROUP_INDEX,
                &self.scene_bind_group,
                &[],
            );
            render_pass.set_vertex_buffer(
                0,
                self.vertex_buffer.slice(..),
            );
            render_pass.draw(
                0..vertices.len() as u32,
                0..1,
            );

            for (effect, batch) in self.effects.iter().zip(&effect_batches) {
                if !batch.is_empty() {
                    render_pass.set_bind_group(
                        SCENE_BIND_GROUP_INDEX,
                        &self.scene_bind_group,
                        &[],
                    );
                    render_pass.set_vertex_buffer(
                        0,
                        self.vertex_buffer.slice(..),
                    );
                    effect.draw(
                        &mut render_pass,
                        batch,
                    );
                }
            }
        }
    }
}

fn push_box_vertices(
    out: &mut Vec<SceneVertex>,
    center: glam::Vec3,
    rotation: glam::Quat,
    half: glam::Vec3,
    base_color: [f32; 3],
) {
    let corners = [
        glam::Vec3::new(
            -half.x, -half.y, -half.z,
        ),
        glam::Vec3::new(
            half.x, -half.y, -half.z,
        ),
        glam::Vec3::new(
            half.x, half.y, -half.z,
        ),
        glam::Vec3::new(
            -half.x, half.y, -half.z,
        ),
        glam::Vec3::new(
            -half.x, -half.y, half.z,
        ),
        glam::Vec3::new(
            half.x, -half.y, half.z,
        ),
        glam::Vec3::new(
            half.x, half.y, half.z,
        ),
        glam::Vec3::new(
            -half.x, half.y, half.z,
        ),
    ]
    .map(|local| (center + (rotation * local)).to_array());

    // Shade faces slightly differently for clear 3D readability without lights in the stub shader.
    let faces: [(
        [usize; 4],
        glam::Vec3,
        f32,
    ); 6] = [
        (
            [4, 5, 6, 7],
            glam::Vec3::Z,
            1.00,
        ), // +Z front
        (
            [1, 0, 3, 2],
            glam::Vec3::NEG_Z,
            0.72,
        ), // -Z back
        (
            [5, 1, 2, 6],
            glam::Vec3::X,
            0.86,
        ), // +X right
        (
            [0, 4, 7, 3],
            glam::Vec3::NEG_X,
            0.80,
        ), // -X left
        (
            [7, 6, 2, 3],
            glam::Vec3::Y,
            1.12,
        ), // +Y top
        (
            [0, 1, 5, 4],
            glam::Vec3::NEG_Y,
            0.55,
        ), // -Y bottom
    ];
    let quad_uvs: [[f32; 2]; 4] = [[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]];

    for (idx, local_normal, shade) in faces {
        let normal = (rotation * local_normal).normalize_or_zero().to_array();
        let color = [
            (base_color[0] * shade).min(1.0),
            (base_color[1] * shade).min(1.0),
            (base_color[2] * shade).min(1.0),
        ];
        let tri = [0, 1, 2, 0, 2, 3];
        for corner_i in tri {
            out.push(
                SceneVertex {
                    position: corners[idx[corner_i]],
                    normal,
                    uv: quad_uvs[corner_i],
                    color,
                },
            );
        }
    }
}
