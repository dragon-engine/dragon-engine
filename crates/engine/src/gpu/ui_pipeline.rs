#![allow(unsafe_code)]

use ash::vk;
use gpu_allocator::MemoryLocation;

use crate::gpu::device::{VulkanDevice, entry_name_cstring};
use crate::gpu::scene_pipeline::FRAMES_IN_FLIGHT;
use crate::gpu::{GpuBuffer, GpuError, GpuImage, vk_err};

/// Private Vulkan 1.3 UI RGBA8 composite pipeline using `core_shader`'s `ui_vs_main`/`ui_fs_main`.
pub(crate) struct UiPipeline {
    descriptor_pool: vk::DescriptorPool,
    set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    sampler: vk::Sampler,
    descriptor_sets: [vk::DescriptorSet; FRAMES_IN_FLIGHT],
    ui_images: [GpuImage; FRAMES_IN_FLIGHT],
    staging_buffers: [GpuBuffer; FRAMES_IN_FLIGHT],
}

impl UiPipeline {
    pub(crate) fn new(
        vk_dev: &mut VulkanDevice,
        color_format: vk::Format,
        extent: vk::Extent2D,
    ) -> Result<Self, GpuError> {
        let bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        ];
        let set_layout = {
            let info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
            // SAFETY: `device` is valid.
            unsafe {
                vk_dev
                    .device
                    .create_descriptor_set_layout(
                        &info, None,
                    )
                    .map_err(
                        |code| {
                            vk_err(
                                "create_descriptor_set_layout",
                                code,
                            )
                        },
                    )?
            }
        };

        let layouts = [set_layout];
        let pipeline_layout = {
            let info = vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts);
            // SAFETY: `set_layout` is valid.
            unsafe {
                vk_dev
                    .device
                    .create_pipeline_layout(
                        &info, None,
                    )
                    .map_err(
                        |code| {
                            vk_err(
                                "create_pipeline_layout",
                                code,
                            )
                        },
                    )?
            }
        };

        let pipeline = build_ui_graphics_pipeline(
            vk_dev,
            pipeline_layout,
            color_format,
        )?;

        let sampler = {
            let info = vk::SamplerCreateInfo::default()
                .mag_filter(vk::Filter::LINEAR)
                .min_filter(vk::Filter::LINEAR)
                .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE);
            // SAFETY: `device` is valid.
            unsafe {
                vk_dev
                    .device
                    .create_sampler(
                        &info, None,
                    )
                    .map_err(
                        |code| {
                            vk_err(
                                "create_sampler",
                                code,
                            )
                        },
                    )?
            }
        };

        let pool_sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::SAMPLED_IMAGE)
                .descriptor_count(FRAMES_IN_FLIGHT as u32),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::SAMPLER)
                .descriptor_count(FRAMES_IN_FLIGHT as u32),
        ];
        let descriptor_pool = {
            let info = vk::DescriptorPoolCreateInfo::default()
                .max_sets(FRAMES_IN_FLIGHT as u32)
                .pool_sizes(&pool_sizes);
            // SAFETY: `device` is valid.
            unsafe {
                vk_dev
                    .device
                    .create_descriptor_pool(
                        &info, None,
                    )
                    .map_err(
                        |code| {
                            vk_err(
                                "create_descriptor_pool",
                                code,
                            )
                        },
                    )?
            }
        };

        let alloc_layouts = [set_layout; FRAMES_IN_FLIGHT];
        let sets_vec = {
            let info = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(descriptor_pool)
                .set_layouts(&alloc_layouts);
            // SAFETY: `descriptor_pool` has capacity for `FRAMES_IN_FLIGHT` sets.
            unsafe {
                vk_dev.device.allocate_descriptor_sets(&info).map_err(
                    |code| {
                        vk_err(
                            "allocate_descriptor_sets",
                            code,
                        )
                    },
                )?
            }
        };
        let descriptor_sets = [sets_vec[0], sets_vec[1]];

        let width = extent.width.max(1);
        let height = extent.height.max(1);
        let ui_images = [
            create_ui_image(
                vk_dev,
                "UI Overlay Image 0",
                width,
                height,
            )?,
            create_ui_image(
                vk_dev,
                "UI Overlay Image 1",
                width,
                height,
            )?,
        ];
        let staging_bytes = u64::from(width) * u64::from(height) * 4;
        let staging_buffers = [
            create_ui_staging(
                vk_dev,
                "UI Staging Buffer 0",
                staging_bytes,
            )?,
            create_ui_staging(
                vk_dev,
                "UI Staging Buffer 1",
                staging_bytes,
            )?,
        ];

        for i in 0..FRAMES_IN_FLIGHT {
            write_ui_descriptor(
                &vk_dev.device,
                descriptor_sets[i],
                ui_images[i].view,
                sampler,
            );
        }

        Ok(
            Self {
                descriptor_pool,
                set_layout,
                pipeline_layout,
                pipeline,
                sampler,
                descriptor_sets,
                ui_images,
                staging_buffers,
            },
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_composite(
        &mut self,
        vk_dev: &mut VulkanDevice,
        cmd: vk::CommandBuffer,
        swapchain_view: vk::ImageView,
        swapchain_extent: vk::Extent2D,
        frame_idx: usize,
        width: u32,
        height: u32,
        rgba8_pixels: &[u8],
    ) {
        let width = width.max(1);
        let height = height.max(1);
        let required_bytes = (width as usize)
            .saturating_mul(height as usize)
            .saturating_mul(4);
        if rgba8_pixels.len() < required_bytes {
            tracing::warn!(
                provided = rgba8_pixels.len(),
                required = required_bytes,
                "UI RGBA8 buffer smaller than target dimensions; skipping UI composite"
            );
            return;
        }

        if self.ui_images[frame_idx].width != width || self.ui_images[frame_idx].height != height {
            // Frame fence already waited for `frame_idx`, so recreating this slot's image/staging buffer is safe.
            let device = vk_dev.device.clone();
            self.ui_images[frame_idx].destroy(
                &device,
                vk_dev.allocator_mut(),
            );
            match create_ui_image(
                vk_dev,
                "UI Overlay Image",
                width,
                height,
            ) {
                Ok(img) => {
                    self.ui_images[frame_idx] = img;
                    write_ui_descriptor(
                        &vk_dev.device,
                        self.descriptor_sets[frame_idx],
                        self.ui_images[frame_idx].view,
                        self.sampler,
                    );
                }
                Err(err) => {
                    tracing::error!(
                        ?err,
                        "Failed to resize UI overlay image"
                    );
                    return;
                }
            }
        }

        if self.staging_buffers[frame_idx].size < required_bytes as u64 {
            let device = vk_dev.device.clone();
            self.staging_buffers[frame_idx].destroy(
                &device,
                vk_dev.allocator_mut(),
            );
            match create_ui_staging(
                vk_dev,
                "UI Staging Buffer",
                required_bytes as u64,
            ) {
                Ok(buf) => self.staging_buffers[frame_idx] = buf,
                Err(err) => {
                    tracing::error!(
                        ?err,
                        "Failed to grow UI staging buffer"
                    );
                    return;
                }
            }
        }

        if let Err(err) = self.staging_buffers[frame_idx].write_bytes(
            0,
            &rgba8_pixels[..required_bytes],
        ) {
            tracing::error!(
                ?err,
                "Failed to stage UI pixels"
            );
            return;
        }

        let ui_image = self.ui_images[frame_idx].image;
        let subresource_range = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        };

        let to_transfer = [vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
            .src_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
            .dst_stage_mask(vk::PipelineStageFlags2::COPY)
            .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .image(ui_image)
            .subresource_range(subresource_range)];
        let dep_to_transfer = vk::DependencyInfo::default().image_memory_barriers(&to_transfer);

        let copy_region = [vk::BufferImageCopy::default()
            .buffer_offset(0)
            .buffer_row_length(0)
            .buffer_image_height(0)
            .image_subresource(
                vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                },
            )
            .image_extent(
                vk::Extent3D {
                    width,
                    height,
                    depth: 1,
                },
            )];

        let to_shader_read = [vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COPY)
            .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
            .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
            .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image(ui_image)
            .subresource_range(subresource_range)];
        let dep_to_read = vk::DependencyInfo::default().image_memory_barriers(&to_shader_read);

        let color_attachment = [vk::RenderingAttachmentInfo::default()
            .image_view(swapchain_view)
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::LOAD)
            .store_op(vk::AttachmentStoreOp::STORE)];
        let rendering_info = vk::RenderingInfo::default()
            .render_area(
                vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: swapchain_extent,
                },
            )
            .layer_count(1)
            .color_attachments(&color_attachment);

        let w = swapchain_extent.width.max(1) as f32;
        let h = swapchain_extent.height.max(1) as f32;
        let viewport = [vk::Viewport {
            x: 0.0,
            y: h,
            width: w,
            height: -h,
            min_depth: 0.0,
            max_depth: 1.0,
        }];
        let scissor = [vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: swapchain_extent,
        }];

        // SAFETY: `cmd` is recording, swapchain image is in `COLOR_ATTACHMENT_OPTIMAL`, and
        // per-frame UI staging buffer/image are owned exclusively by this frame slot.
        unsafe {
            vk_dev.device.cmd_pipeline_barrier2(
                cmd,
                &dep_to_transfer,
            );
            vk_dev.device.cmd_copy_buffer_to_image(
                cmd,
                self.staging_buffers[frame_idx].buffer,
                ui_image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &copy_region,
            );
            vk_dev.device.cmd_pipeline_barrier2(
                cmd,
                &dep_to_read,
            );

            vk_dev.device.cmd_begin_rendering(
                cmd,
                &rendering_info,
            );
            vk_dev.device.cmd_set_viewport(
                cmd, 0, &viewport,
            );
            vk_dev.device.cmd_set_scissor(
                cmd, 0, &scissor,
            );
            vk_dev.device.cmd_bind_pipeline(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline,
            );
            vk_dev.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout,
                0,
                &[self.descriptor_sets[frame_idx]],
                &[],
            );
            vk_dev.device.cmd_draw(
                cmd, 3, 1, 0, 0,
            );
            vk_dev.device.cmd_end_rendering(cmd);
        }
    }

    pub(crate) fn destroy(&mut self, vk_dev: &mut VulkanDevice) {
        let device = vk_dev.device.clone();
        let allocator = vk_dev.allocator_mut();
        for img in &mut self.ui_images {
            img.destroy(
                &device, allocator,
            );
        }
        for buf in &mut self.staging_buffers {
            buf.destroy(
                &device, allocator,
            );
        }
        // SAFETY: Device is idle before destruction.
        unsafe {
            device.destroy_sampler(
                self.sampler,
                None,
            );
            device.destroy_pipeline(
                self.pipeline,
                None,
            );
            device.destroy_pipeline_layout(
                self.pipeline_layout,
                None,
            );
            device.destroy_descriptor_pool(
                self.descriptor_pool,
                None,
            );
            device.destroy_descriptor_set_layout(
                self.set_layout,
                None,
            );
        }
    }
}

fn create_ui_image(
    vk_dev: &mut VulkanDevice,
    name: &str,
    width: u32,
    height: u32,
) -> Result<GpuImage, GpuError> {
    let device = vk_dev.device.clone();
    GpuImage::create_2d(
        &device,
        vk_dev.allocator_mut(),
        name,
        width,
        height,
        vk::Format::R8G8B8A8_SRGB,
        vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
        vk::ImageAspectFlags::COLOR,
    )
}

fn create_ui_staging(
    vk_dev: &mut VulkanDevice,
    name: &str,
    bytes: u64,
) -> Result<GpuBuffer, GpuError> {
    let device = vk_dev.device.clone();
    GpuBuffer::create(
        &device,
        vk_dev.allocator_mut(),
        name,
        bytes,
        vk::BufferUsageFlags::TRANSFER_SRC,
        MemoryLocation::CpuToGpu,
    )
}

fn write_ui_descriptor(
    device: &ash::Device,
    set: vk::DescriptorSet,
    view: vk::ImageView,
    sampler: vk::Sampler,
) {
    let img_info = [vk::DescriptorImageInfo::default()
        .image_view(view)
        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
    let sampler_info = [vk::DescriptorImageInfo::default().sampler(sampler)];
    let writes = [
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
            .image_info(&img_info),
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(1)
            .descriptor_type(vk::DescriptorType::SAMPLER)
            .image_info(&sampler_info),
    ];
    // SAFETY: `set`, `view`, and `sampler` are valid and unused by the GPU.
    unsafe {
        device.update_descriptor_sets(
            &writes,
            &[],
        );
    }
}

fn build_ui_graphics_pipeline(
    vk_dev: &VulkanDevice,
    layout: vk::PipelineLayout,
    color_format: vk::Format,
) -> Result<vk::Pipeline, GpuError> {
    let module = vk_dev.create_shader_module(include_bytes!(env!("CORE_SHADER_SPV")))?;
    let vs_entry = entry_name_cstring("ui_vs_main")?;
    let fs_entry = entry_name_cstring("ui_fs_main")?;

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

    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
    let viewport_state = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        .scissor_count(1);
    let rasterizer = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(vk::CullModeFlags::NONE)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .line_width(1.0);
    let multisampling = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);
    let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
        .depth_test_enable(false)
        .depth_write_enable(false);

    // Premultiplied alpha blending matching `iced_tiny_skia` RGBA8 output.
    let blend_attachments = [vk::PipelineColorBlendAttachmentState::default()
        .blend_enable(true)
        .src_color_blend_factor(vk::BlendFactor::ONE)
        .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
        .color_blend_op(vk::BlendOp::ADD)
        .src_alpha_blend_factor(vk::BlendFactor::ONE)
        .dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
        .alpha_blend_op(vk::BlendOp::ADD)
        .color_write_mask(vk::ColorComponentFlags::RGBA)];
    let color_blending =
        vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend_attachments);

    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic_state =
        vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

    let color_formats = [color_format];
    let mut rendering_info =
        vk::PipelineRenderingCreateInfo::default().color_attachment_formats(&color_formats);

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
            .map_err(
                |(_, code)| {
                    vk_err(
                        "create_graphics_pipelines",
                        code,
                    )
                },
            )
    };

    // SAFETY: Shader module may be destroyed immediately after pipeline creation.
    unsafe {
        vk_dev.device.destroy_shader_module(
            module, None,
        );
    }

    result.map(|pipelines| pipelines[0])
}
