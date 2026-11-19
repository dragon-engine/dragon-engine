#![allow(unsafe_code)]

use std::mem::offset_of;

use ash::vk;
use gpu_allocator::MemoryLocation;
use shader_abi::{
    EFFECT_BIND_GROUP_INDEX, Mat4f, SCENE_BIND_GROUP_INDEX, SceneUniforms, SceneVertex, Vec4f,
};

use crate::effect::{EffectBlendMode, EffectDrawItem, EffectRegistry, EffectShaderSpec};
use crate::gpu::device::{VulkanDevice, entry_name_cstring};
use crate::gpu::{GpuBuffer, GpuError, GpuImage, vk_err};
use crate::snapshot::RenderSnapshot;

pub(crate) const FRAMES_IN_FLIGHT: usize = 2;
const MAX_VERTICES: usize = 36 * 2048;
const PARAM_BYTES: usize = size_of::<[[f32; 4]; 2]>();
const MAX_EFFECT_ITEMS: usize = 512;
const DEPTH_FORMAT: vk::Format = vk::Format::D32_SFLOAT;

struct CompiledEffect {
    pipeline: vk::Pipeline,
    param_buffers: [GpuBuffer; FRAMES_IN_FLIGHT],
    descriptor_sets: [vk::DescriptorSet; FRAMES_IN_FLIGHT],
}

/// Private Vulkan 1.3 dynamic-rendering scene and pluggable effect pipeline.
pub(crate) struct ScenePipeline {
    descriptor_pool: vk::DescriptorPool,
    scene_set_layout: vk::DescriptorSetLayout,
    effect_set_layout: vk::DescriptorSetLayout,
    scene_pipeline_layout: vk::PipelineLayout,
    effect_pipeline_layout: vk::PipelineLayout,
    base_pipeline: vk::Pipeline,
    scene_ubos: [GpuBuffer; FRAMES_IN_FLIGHT],
    scene_sets: [vk::DescriptorSet; FRAMES_IN_FLIGHT],
    vertex_buffers: [GpuBuffer; FRAMES_IN_FLIGHT],
    depth_image: GpuImage,
    effects: Vec<CompiledEffect>,
    param_stride: usize,
}

impl ScenePipeline {
    pub(crate) fn new(
        vk_dev: &mut VulkanDevice,
        color_format: vk::Format,
        extent: vk::Extent2D,
        effects: EffectRegistry,
    ) -> Result<Self, GpuError> {
        let registered = effects.into_effects();
        let min_align = (vk_dev.capabilities.min_uniform_buffer_offset_alignment as usize).max(32);
        let param_stride = PARAM_BYTES.next_multiple_of(min_align);

        let scene_bindings = [vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)];
        let scene_set_layout = {
            let info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&scene_bindings);
            // SAFETY: `device` is valid.
            unsafe {
                vk_dev
                    .device
                    .create_descriptor_set_layout(
                        &info, None,
                    )
                    .map_err(|code| vk_err("create_descriptor_set_layout", code))?
            }
        };

        let effect_bindings = [vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)];
        let effect_set_layout = {
            let info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&effect_bindings);
            // SAFETY: `device` is valid.
            unsafe {
                vk_dev
                    .device
                    .create_descriptor_set_layout(
                        &info, None,
                    )
                    .map_err(|code| vk_err("create_descriptor_set_layout", code))?
            }
        };

        let scene_layouts = [scene_set_layout];
        let scene_pipeline_layout = {
            let info = vk::PipelineLayoutCreateInfo::default().set_layouts(&scene_layouts);
            // SAFETY: `scene_set_layout` is valid.
            unsafe {
                vk_dev
                    .device
                    .create_pipeline_layout(
                        &info, None,
                    )
                    .map_err(|code| vk_err("create_pipeline_layout", code))?
            }
        };

        let effect_layouts = [scene_set_layout, effect_set_layout];
        let effect_pipeline_layout = {
            let info = vk::PipelineLayoutCreateInfo::default().set_layouts(&effect_layouts);
            // SAFETY: Both descriptor set layouts are valid.
            unsafe {
                vk_dev
                    .device
                    .create_pipeline_layout(
                        &info, None,
                    )
                    .map_err(|code| vk_err("create_pipeline_layout", code))?
            }
        };

        let base_spec = EffectShaderSpec {
            label: "Core Scene Shader",
            spirv_bytes: include_bytes!(env!("CORE_SHADER_SPV")),
            vertex_entry: "vs_main",
            fragment_entry: "fs_main",
            blend: EffectBlendMode::Opaque,
        };
        let base_pipeline = build_mesh_graphics_pipeline(
            vk_dev,
            scene_pipeline_layout,
            color_format,
            base_spec,
            true,
            vk::CompareOp::LESS,
        )?;

        let effect_set_count = (registered.len() * FRAMES_IN_FLIGHT).max(1) as u32;
        let pool_sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(FRAMES_IN_FLIGHT as u32),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC)
                .descriptor_count(effect_set_count),
        ];
        let descriptor_pool = {
            let info = vk::DescriptorPoolCreateInfo::default()
                .max_sets(FRAMES_IN_FLIGHT as u32 + effect_set_count)
                .pool_sizes(&pool_sizes);
            // SAFETY: `device` is valid.
            unsafe {
                vk_dev
                    .device
                    .create_descriptor_pool(
                        &info, None,
                    )
                    .map_err(|code| vk_err("create_descriptor_pool", code))?
            }
        };

        let alloc_scene_layouts = [scene_set_layout; FRAMES_IN_FLIGHT];
        let scene_sets_vec = {
            let info = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(descriptor_pool)
                .set_layouts(&alloc_scene_layouts);
            // SAFETY: `descriptor_pool` has capacity for `FRAMES_IN_FLIGHT` UBO sets.
            unsafe {
                vk_dev
                    .device
                    .allocate_descriptor_sets(&info)
                    .map_err(|code| vk_err("allocate_descriptor_sets", code))?
            }
        };
        let scene_sets = [scene_sets_vec[0], scene_sets_vec[1]];

        let scene_ubos = [
            create_scene_ubo(
                vk_dev,
                "Scene UBO 0",
            )?,
            create_scene_ubo(
                vk_dev,
                "Scene UBO 1",
            )?,
        ];
        for (i, ubo) in scene_ubos.iter().enumerate() {
            let buf_info = [vk::DescriptorBufferInfo::default()
                .buffer(ubo.buffer)
                .offset(0)
                .range(size_of::<SceneUniforms>() as u64)];
            let write = vk::WriteDescriptorSet::default()
                .dst_set(scene_sets[i])
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(&buf_info);
            // SAFETY: `scene_sets[i]` and `ubo.buffer` are valid and unused.
            unsafe {
                vk_dev.device.update_descriptor_sets(
                    &[write],
                    &[],
                );
            }
        }

        let vertex_bytes = (MAX_VERTICES * size_of::<SceneVertex>()) as u64;
        let vertex_buffers = [
            GpuBuffer::create(
                &vk_dev.device.clone(),
                vk_dev.allocator_mut(),
                "Scene Vertex Buffer 0",
                vertex_bytes,
                vk::BufferUsageFlags::VERTEX_BUFFER,
                MemoryLocation::CpuToGpu,
            )?,
            GpuBuffer::create(
                &vk_dev.device.clone(),
                vk_dev.allocator_mut(),
                "Scene Vertex Buffer 1",
                vertex_bytes,
                vk::BufferUsageFlags::VERTEX_BUFFER,
                MemoryLocation::CpuToGpu,
            )?,
        ];

        let depth_image = create_depth_image(
            vk_dev,
            extent.width,
            extent.height,
        )?;

        let mut compiled_effects = Vec::with_capacity(registered.len());
        let param_buf_bytes = (param_stride * MAX_EFFECT_ITEMS) as u64;
        for effect in registered {
            let spec = effect.shader_spec();
            let pipeline = build_mesh_graphics_pipeline(
                vk_dev,
                effect_pipeline_layout,
                color_format,
                spec,
                false,
                vk::CompareOp::LESS_OR_EQUAL,
            )?;
            let alloc_layouts = [effect_set_layout; FRAMES_IN_FLIGHT];
            let sets_vec = {
                let info = vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(descriptor_pool)
                    .set_layouts(&alloc_layouts);
                // SAFETY: `descriptor_pool` was sized for all registered effects.
                unsafe {
                    vk_dev
                        .device
                        .allocate_descriptor_sets(&info)
                        .map_err(|code| vk_err("allocate_descriptor_sets", code))?
                }
            };
            let descriptor_sets = [sets_vec[0], sets_vec[1]];
            let param_buffers = [
                GpuBuffer::create(
                    &vk_dev.device.clone(),
                    vk_dev.allocator_mut(),
                    "Effect Param UBO 0",
                    param_buf_bytes,
                    vk::BufferUsageFlags::UNIFORM_BUFFER,
                    MemoryLocation::CpuToGpu,
                )?,
                GpuBuffer::create(
                    &vk_dev.device.clone(),
                    vk_dev.allocator_mut(),
                    "Effect Param UBO 1",
                    param_buf_bytes,
                    vk::BufferUsageFlags::UNIFORM_BUFFER,
                    MemoryLocation::CpuToGpu,
                )?,
            ];
            for (i, buf) in param_buffers.iter().enumerate() {
                let buf_info = [vk::DescriptorBufferInfo::default()
                    .buffer(buf.buffer)
                    .offset(0)
                    .range(PARAM_BYTES as u64)];
                let write = vk::WriteDescriptorSet::default()
                    .dst_set(descriptor_sets[i])
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC)
                    .buffer_info(&buf_info);
                // SAFETY: `descriptor_sets[i]` and `buf.buffer` are valid and unused.
                unsafe {
                    vk_dev.device.update_descriptor_sets(
                        &[write],
                        &[],
                    );
                }
            }
            compiled_effects.push(
                CompiledEffect {
                    pipeline,
                    param_buffers,
                    descriptor_sets,
                },
            );
        }

        Ok(
            Self {
                descriptor_pool,
                scene_set_layout,
                effect_set_layout,
                scene_pipeline_layout,
                effect_pipeline_layout,
                base_pipeline,
                scene_ubos,
                scene_sets,
                vertex_buffers,
                depth_image,
                effects: compiled_effects,
                param_stride,
            },
        )
    }

    pub(crate) fn depth_image(&self) -> vk::Image {
        self.depth_image.image
    }

    pub(crate) fn resize(
        &mut self,
        vk_dev: &mut VulkanDevice,
        width: u32,
        height: u32,
    ) -> Result<(), GpuError> {
        let device = vk_dev.device.clone();
        self.depth_image.destroy(
            &device,
            vk_dev.allocator_mut(),
        );
        self.depth_image = create_depth_image(
            vk_dev, width, height,
        )?;
        Ok(())
    }

    pub(crate) fn record<G>(
        &mut self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        color_view: vk::ImageView,
        extent: vk::Extent2D,
        frame_idx: usize,
        snapshot: &RenderSnapshot<G>,
    ) {
        let width = extent.width.max(1) as f32;
        let height = extent.height.max(1) as f32;
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
        if let Err(err) = self.scene_ubos[frame_idx].write_bytes(
            0,
            bytemuck::bytes_of(&uniforms),
        ) {
            tracing::error!(?err, "Failed to update scene uniforms");
        }

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
                && batch.len() < MAX_EFFECT_ITEMS
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
            if let Err(err) = self.vertex_buffers[frame_idx].write_bytes(
                0,
                bytemuck::cast_slice(&vertices),
            ) {
                tracing::error!(?err, "Failed to upload scene vertices");
            }
            for (effect, batch) in self.effects.iter_mut().zip(&effect_batches) {
                for (i, item) in batch.iter().enumerate() {
                    let offset = i * self.param_stride;
                    if let Err(err) = effect.param_buffers[frame_idx].write_bytes(
                        offset,
                        bytemuck::bytes_of(&item.params),
                    ) {
                        tracing::error!(?err, "Failed to upload effect uniform parameters");
                    }
                }
            }
        }

        let color_attachment = [vk::RenderingAttachmentInfo::default()
            .image_view(color_view)
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(
                vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: [0.08, 0.11, 0.16, 1.0],
                    },
                },
            )];
        let depth_attachment = vk::RenderingAttachmentInfo::default()
            .image_view(self.depth_image.view)
            .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(
                vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue {
                        depth: 1.0,
                        stencil: 0,
                    },
                },
            );
        let rendering_info = vk::RenderingInfo::default()
            .render_area(
                vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent,
                },
            )
            .layer_count(1)
            .color_attachments(&color_attachment)
            .depth_attachment(&depth_attachment);

        // Negative Vulkan viewport height flips NDC Y so right-handed +Y up and CCW winding
        // match `glam::camera::rh::proj::directx::perspective` without modifying shader math.
        let viewport = [vk::Viewport {
            x: 0.0,
            y: height,
            width,
            height: -height,
            min_depth: 0.0,
            max_depth: 1.0,
        }];
        let scissor = [vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent,
        }];

        // SAFETY: `cmd` is recording, attachments are transitioned, and buffers are live for this frame.
        unsafe {
            device.cmd_begin_rendering(
                cmd,
                &rendering_info,
            );
            device.cmd_set_viewport(
                cmd, 0, &viewport,
            );
            device.cmd_set_scissor(
                cmd, 0, &scissor,
            );

            if !vertices.is_empty() {
                device.cmd_bind_pipeline(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.base_pipeline,
                );
                device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.scene_pipeline_layout,
                    SCENE_BIND_GROUP_INDEX,
                    &[self.scene_sets[frame_idx]],
                    &[],
                );
                device.cmd_bind_vertex_buffers(
                    cmd,
                    0,
                    &[self.vertex_buffers[frame_idx].buffer],
                    &[0],
                );
                device.cmd_draw(
                    cmd,
                    vertices.len() as u32,
                    1,
                    0,
                    0,
                );

                for (effect, batch) in self.effects.iter().zip(&effect_batches) {
                    if batch.is_empty() {
                        continue;
                    }
                    device.cmd_bind_pipeline(
                        cmd,
                        vk::PipelineBindPoint::GRAPHICS,
                        effect.pipeline,
                    );
                    device.cmd_bind_descriptor_sets(
                        cmd,
                        vk::PipelineBindPoint::GRAPHICS,
                        self.effect_pipeline_layout,
                        SCENE_BIND_GROUP_INDEX,
                        &[self.scene_sets[frame_idx]],
                        &[],
                    );
                    device.cmd_bind_vertex_buffers(
                        cmd,
                        0,
                        &[self.vertex_buffers[frame_idx].buffer],
                        &[0],
                    );
                    for (i, item) in batch.iter().enumerate() {
                        let dynamic_offset = u32::try_from(i * self.param_stride)
                            .unwrap_or_else(|err| panic!("Uniform offset overflow: {err}"));
                        device.cmd_bind_descriptor_sets(
                            cmd,
                            vk::PipelineBindPoint::GRAPHICS,
                            self.effect_pipeline_layout,
                            EFFECT_BIND_GROUP_INDEX,
                            &[effect.descriptor_sets[frame_idx]],
                            &[dynamic_offset],
                        );
                        device.cmd_draw(
                            cmd,
                            item.vertex_range.end - item.vertex_range.start,
                            1,
                            item.vertex_range.start,
                            0,
                        );
                    }
                }
            }

            device.cmd_end_rendering(cmd);
        }
    }

    pub(crate) fn destroy(&mut self, vk_dev: &mut VulkanDevice) {
        let device = vk_dev.device.clone();
        let allocator = vk_dev.allocator_mut();
        self.depth_image.destroy(
            &device, allocator,
        );
        for buf in &mut self.scene_ubos {
            buf.destroy(
                &device, allocator,
            );
        }
        for buf in &mut self.vertex_buffers {
            buf.destroy(
                &device, allocator,
            );
        }
        for effect in &mut self.effects {
            for buf in &mut effect.param_buffers {
                buf.destroy(
                    &device, allocator,
                );
            }
            // SAFETY: Device is idle before destruction.
            unsafe {
                device.destroy_pipeline(
                    effect.pipeline,
                    None,
                );
            }
        }
        self.effects.clear();
        // SAFETY: Device is idle and child resources are destroyed.
        unsafe {
            device.destroy_pipeline(
                self.base_pipeline,
                None,
            );
            device.destroy_pipeline_layout(
                self.effect_pipeline_layout,
                None,
            );
            device.destroy_pipeline_layout(
                self.scene_pipeline_layout,
                None,
            );
            device.destroy_descriptor_pool(
                self.descriptor_pool,
                None,
            );
            device.destroy_descriptor_set_layout(
                self.effect_set_layout,
                None,
            );
            device.destroy_descriptor_set_layout(
                self.scene_set_layout,
                None,
            );
        }
    }
}

/// Validates that an `EffectShaderSpec` compiles into a Vulkan 1.3 dynamic-rendering pipeline.
/// Skips hardware pipeline creation when no Vulkan 1.3 ICD is present on the host, while always
/// validating SPIR-V header/alignment.
pub(crate) fn validate_effect_shader_headless(spec: EffectShaderSpec) -> Result<(), GpuError> {
    let _ = crate::gpu::decode_spirv_words(spec.spirv_bytes)?;
    let vk_dev = match VulkanDevice::new_headless() {
        Ok(dev) => dev,
        Err(GpuError::Loader(msg) | GpuError::UnsupportedAdapter(msg)) => {
            tracing::warn!(%msg, "Skipping Vulkan pipeline compilation test on headless host without Vulkan 1.3 ICD");
            return Ok(());
        }
        Err(err) => return Err(err),
    };

    let scene_bindings = [vk::DescriptorSetLayoutBinding::default()
        .binding(0)
        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)];
    let effect_bindings = [vk::DescriptorSetLayoutBinding::default()
        .binding(0)
        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::FRAGMENT)];

    // SAFETY: `vk_dev.device` is valid and handles are destroyed before returning.
    let scene_layout = unsafe {
        vk_dev
            .device
            .create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&scene_bindings),
                None,
            )
            .map_err(|code| vk_err("create_descriptor_set_layout", code))?
    };
    let effect_layout = unsafe {
        vk_dev
            .device
            .create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&effect_bindings),
                None,
            )
            .map_err(|code| vk_err("create_descriptor_set_layout", code))?
    };
    let layouts = [scene_layout, effect_layout];
    let pipeline_layout = unsafe {
        vk_dev
            .device
            .create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts),
                None,
            )
            .map_err(|code| vk_err("create_pipeline_layout", code))?
    };

    let pipeline_res = build_mesh_graphics_pipeline(
        &vk_dev,
        pipeline_layout,
        vk::Format::B8G8R8A8_SRGB,
        spec,
        false,
        vk::CompareOp::LESS_OR_EQUAL,
    );

    // SAFETY: Clean up all temporary validation handles before `vk_dev` drops.
    unsafe {
        if let Ok(pipeline) = pipeline_res {
            vk_dev.device.destroy_pipeline(
                pipeline, None,
            );
        }
        vk_dev.device.destroy_pipeline_layout(
            pipeline_layout,
            None,
        );
        vk_dev.device.destroy_descriptor_set_layout(
            effect_layout,
            None,
        );
        vk_dev.device.destroy_descriptor_set_layout(
            scene_layout,
            None,
        );
    }

    pipeline_res.map(|_| ())
}

fn create_scene_ubo(vk_dev: &mut VulkanDevice, name: &str) -> Result<GpuBuffer, GpuError> {
    let device = vk_dev.device.clone();
    GpuBuffer::create(
        &device,
        vk_dev.allocator_mut(),
        name,
        size_of::<SceneUniforms>() as u64,
        vk::BufferUsageFlags::UNIFORM_BUFFER,
        MemoryLocation::CpuToGpu,
    )
}

fn create_depth_image(
    vk_dev: &mut VulkanDevice,
    width: u32,
    height: u32,
) -> Result<GpuImage, GpuError> {
    let device = vk_dev.device.clone();
    GpuImage::create_2d(
        &device,
        vk_dev.allocator_mut(),
        "Scene Depth Image",
        width,
        height,
        DEPTH_FORMAT,
        vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
        vk::ImageAspectFlags::DEPTH,
    )
}

fn build_mesh_graphics_pipeline(
    vk_dev: &VulkanDevice,
    layout: vk::PipelineLayout,
    color_format: vk::Format,
    spec: EffectShaderSpec,
    depth_write: bool,
    depth_compare: vk::CompareOp,
) -> Result<vk::Pipeline, GpuError> {
    let module = vk_dev.create_shader_module(spec.spirv_bytes)?;
    let vs_entry = entry_name_cstring(spec.vertex_entry)?;
    let fs_entry = entry_name_cstring(spec.fragment_entry)?;

    let stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(module)
            .name(&vs_entry),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(module)
            .name(&fs_entry),
    ];

    let vertex_bindings = [vk::VertexInputBindingDescription::default()
        .binding(0)
        .stride(size_of::<SceneVertex>() as u32)
        .input_rate(vk::VertexInputRate::VERTEX)];
    let vertex_attributes = [
        vk::VertexInputAttributeDescription::default()
            .location(0)
            .binding(0)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(offset_of!(SceneVertex, position) as u32),
        vk::VertexInputAttributeDescription::default()
            .location(1)
            .binding(0)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(offset_of!(SceneVertex, normal) as u32),
        vk::VertexInputAttributeDescription::default()
            .location(2)
            .binding(0)
            .format(vk::Format::R32G32_SFLOAT)
            .offset(offset_of!(SceneVertex, uv) as u32),
        vk::VertexInputAttributeDescription::default()
            .location(3)
            .binding(0)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(offset_of!(SceneVertex, color) as u32),
    ];
    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
        .vertex_binding_descriptions(&vertex_bindings)
        .vertex_attribute_descriptions(&vertex_attributes);

    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);

    let viewport_state = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        .scissor_count(1);

    let rasterizer = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(vk::CullModeFlags::BACK)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .line_width(1.0);

    let multisampling = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);

    let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
        .depth_test_enable(true)
        .depth_write_enable(depth_write)
        .depth_compare_op(depth_compare);

    let blend_attachment = match spec.blend {
        EffectBlendMode::Opaque => vk::PipelineColorBlendAttachmentState::default()
            .blend_enable(false)
            .color_write_mask(vk::ColorComponentFlags::RGBA),
        EffectBlendMode::AlphaBlend => vk::PipelineColorBlendAttachmentState::default()
            .blend_enable(true)
            .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
            .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .color_blend_op(vk::BlendOp::ADD)
            .src_alpha_blend_factor(vk::BlendFactor::ONE)
            .dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .alpha_blend_op(vk::BlendOp::ADD)
            .color_write_mask(vk::ColorComponentFlags::RGBA),
        EffectBlendMode::Additive => vk::PipelineColorBlendAttachmentState::default()
            .blend_enable(true)
            .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
            .dst_color_blend_factor(vk::BlendFactor::ONE)
            .color_blend_op(vk::BlendOp::ADD)
            .src_alpha_blend_factor(vk::BlendFactor::ONE)
            .dst_alpha_blend_factor(vk::BlendFactor::ONE)
            .alpha_blend_op(vk::BlendOp::ADD)
            .color_write_mask(vk::ColorComponentFlags::RGBA),
    };
    let blend_attachments = [blend_attachment];
    let color_blending =
        vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend_attachments);

    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic_state =
        vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

    let color_formats = [color_format];
    let mut rendering_info = vk::PipelineRenderingCreateInfo::default()
        .color_attachment_formats(&color_formats)
        .depth_attachment_format(DEPTH_FORMAT);

    let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&stages)
        .vertex_input_state(&vertex_input)
        .input_assembly_state(&input_assembly)
        .viewport_state(&viewport_state)
        .rasterization_state(&rasterizer)
        .multisample_state(&multisampling)
        .depth_stencil_state(&depth_stencil)
        .color_blend_state(&color_blending)
        .dynamic_state(&dynamic_state)
        .layout(layout)
        .push_next(&mut rendering_info);

    // SAFETY: All pipeline create info structs and shader modules are valid for this call.
    let result = unsafe {
        vk_dev
            .device
            .create_graphics_pipelines(
                vk::PipelineCache::null(),
                &[pipeline_info],
                None,
            )
            .map_err(|(_, code)| vk_err("create_graphics_pipelines", code))
    };

    // SAFETY: Shader module may be destroyed immediately after pipeline creation.
    unsafe {
        vk_dev.device.destroy_shader_module(
            module, None,
        );
    }

    result.map(|pipelines| pipelines[0])
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
