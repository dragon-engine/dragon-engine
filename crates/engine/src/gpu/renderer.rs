#![allow(unsafe_code)]

use std::sync::Arc;

use ash::vk;
use winit::dpi::PhysicalSize;
use winit::window::Window;

use crate::effect::EffectRegistry;
use crate::gpu::device::VulkanDevice;
use crate::gpu::scene_pipeline::{FRAMES_IN_FLIGHT, ScenePipeline};
use crate::gpu::swapchain::VulkanSwapchain;
use crate::gpu::ui_pipeline::UiPipeline;
use crate::gpu::{GpuError, vk_err};
use crate::runner::FramePacing;
use crate::snapshot::RenderSnapshot;
use crate::ui::{UiCompositor, UiOverlay, UiRenderTarget};

struct FrameSync {
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    image_available: vk::Semaphore,
    in_flight: vk::Fence,
}

/// Private Vulkan 1.3 renderer owning device, swapchain, frame sync, scene pipeline, and UI pass.
pub(crate) struct VulkanRenderer {
    window: Arc<Window>,
    frame_pacing: FramePacing,
    frames: [FrameSync; FRAMES_IN_FLIGHT],
    // Per-swapchain-image present semaphores prevent reuse while an image is still presented.
    render_finished: Vec<vk::Semaphore>,
    frame_cursor: usize,
    ui_pipeline: UiPipeline,
    scene_pipeline: ScenePipeline,
    swapchain: VulkanSwapchain,
    // `vk_dev` MUST remain the final field so its Drop destroys device/instance last.
    vk_dev: VulkanDevice,
}

struct ActiveUiCompositor<'a> {
    vk_dev: &'a mut VulkanDevice,
    ui_pipeline: &'a mut UiPipeline,
    cmd: vk::CommandBuffer,
    swapchain_view: vk::ImageView,
    swapchain_extent: vk::Extent2D,
    frame_idx: usize,
}

impl UiCompositor for ActiveUiCompositor<'_> {
    fn composite_rgba8(&mut self, width: u32, height: u32, rgba8_pixels: &[u8]) {
        self.ui_pipeline.record_composite(
            self.vk_dev,
            self.cmd,
            self.swapchain_view,
            self.swapchain_extent,
            self.frame_idx,
            width,
            height,
            rgba8_pixels,
        );
    }
}

impl VulkanRenderer {
    pub(crate) fn new(
        window: Arc<Window>,
        size: PhysicalSize<u32>,
        effects: EffectRegistry,
        frame_pacing: FramePacing,
    ) -> Result<Self, GpuError> {
        let (mut vk_dev, surface_loader, surface) = VulkanDevice::new_windowed(&window)?;
        let mut swapchain = VulkanSwapchain::new(
            &vk_dev,
            surface_loader,
            surface,
            size.width,
            size.height,
            frame_pacing,
        )?;
        let scene_pipeline = match ScenePipeline::new(
            &mut vk_dev,
            swapchain.format,
            swapchain.extent,
            effects,
        ) {
            Ok(p) => p,
            Err(err) => {
                swapchain.destroy(&vk_dev.device);
                return Err(err);
            }
        };
        let ui_pipeline = match UiPipeline::new(
            &mut vk_dev,
            swapchain.format,
            swapchain.extent,
        ) {
            Ok(p) => p,
            Err(err) => {
                let mut sp = scene_pipeline;
                sp.destroy(&mut vk_dev);
                swapchain.destroy(&vk_dev.device);
                return Err(err);
            }
        };

        let frames = [create_frame_sync(&vk_dev)?, create_frame_sync(&vk_dev)?];
        let render_finished = create_present_semaphores(
            &vk_dev.device,
            swapchain.images.len(),
        )?;

        Ok(
            Self {
                window,
                frame_pacing,
                frames,
                render_finished,
                frame_cursor: 0,
                ui_pipeline,
                scene_pipeline,
                swapchain,
                vk_dev,
            },
        )
    }

    pub(crate) fn resize(&mut self, size: PhysicalSize<u32>) {
        if size.width == 0 || size.height == 0 {
            return;
        }
        if let Err(err) = self.recreate_swapchain(
            size.width,
            size.height,
        ) {
            tracing::error!(
                ?err,
                "Failed to recreate Vulkan swapchain on resize"
            );
        }
    }

    fn recreate_swapchain(&mut self, width: u32, height: u32) -> Result<(), GpuError> {
        self.swapchain.recreate(
            &self.vk_dev,
            width,
            height,
        )?;
        self.scene_pipeline.resize(
            &mut self.vk_dev,
            self.swapchain.extent.width,
            self.swapchain.extent.height,
        )?;
        if self.render_finished.len() != self.swapchain.images.len() {
            for sem in self.render_finished.drain(..) {
                // SAFETY: `swapchain.recreate` waited for device idle.
                unsafe {
                    self.vk_dev.device.destroy_semaphore(
                        sem, None,
                    );
                }
            }
            self.render_finished = create_present_semaphores(
                &self.vk_dev.device,
                self.swapchain.images.len(),
            )?;
        }
        Ok(())
    }

    pub(crate) fn draw<G, A, U: UiOverlay<G, A>>(
        &mut self,
        snapshot: &RenderSnapshot<G>,
        ui: &mut U,
    ) {
        let frame_idx = self.frame_cursor % FRAMES_IN_FLIGHT;
        let fence = self.frames[frame_idx].in_flight;
        let image_available = self.frames[frame_idx].image_available;

        // SAFETY: `fence` is valid and owned by this frame slot.
        if let Err(code) = unsafe {
            self.vk_dev.device.wait_for_fences(
                &[fence],
                true,
                u64::MAX,
            )
        } {
            tracing::error!(
                ?code,
                "vkWaitForFences failed"
            );
            return;
        }

        // SAFETY: `swapchain` and `image_available` are valid.
        let acquire_res = unsafe {
            self.swapchain.swapchain_loader.acquire_next_image(
                self.swapchain.swapchain,
                u64::MAX,
                image_available,
                vk::Fence::null(),
            )
        };
        let (image_index, suboptimal) = match acquire_res {
            Ok(pair) => pair,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                let size = self.window.inner_size();
                if size.width > 0 && size.height > 0 {
                    self.resize(size);
                }
                return;
            }
            Err(code) => {
                tracing::error!(
                    ?code,
                    "vkAcquireNextImageKHR failed"
                );
                return;
            }
        };

        let image_idx = image_index as usize;
        let render_finished = self.render_finished[image_idx];
        let swapchain_image = self.swapchain.images[image_idx];
        let swapchain_view = self.swapchain.image_views[image_idx];
        let extent = self.swapchain.extent;
        let cmd_pool = self.frames[frame_idx].command_pool;
        let cmd = self.frames[frame_idx].command_buffer;

        // SAFETY: Fence wait completed above, so this frame's command pool and fence are idle.
        unsafe {
            if let Err(code) = self.vk_dev.device.reset_fences(&[fence]) {
                tracing::error!(
                    ?code,
                    "vkResetFences failed"
                );
                return;
            }
            if let Err(code) = self.vk_dev.device.reset_command_pool(
                cmd_pool,
                vk::CommandPoolResetFlags::empty(),
            ) {
                tracing::error!(
                    ?code,
                    "vkResetCommandPool failed"
                );
                return;
            }
            let begin_info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            if let Err(code) = self.vk_dev.device.begin_command_buffer(
                cmd,
                &begin_info,
            ) {
                tracing::error!(
                    ?code,
                    "vkBeginCommandBuffer failed"
                );
                return;
            }
        }

        let color_subresource = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        };
        let depth_subresource = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::DEPTH,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        };
        let pre_render_barriers = [
            vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                .src_access_mask(vk::AccessFlags2::empty())
                .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                .dst_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .image(swapchain_image)
                .subresource_range(color_subresource),
            vk::ImageMemoryBarrier2::default()
                .src_stage_mask(
                    vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                        | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
                )
                .src_access_mask(vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE)
                .dst_stage_mask(
                    vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                        | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
                )
                .dst_access_mask(
                    vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ
                        | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
                )
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                .image(self.scene_pipeline.depth_image())
                .subresource_range(depth_subresource),
        ];
        let pre_dep = vk::DependencyInfo::default().image_memory_barriers(&pre_render_barriers);
        // SAFETY: `cmd` is recording and both images are owned by this frame.
        unsafe {
            self.vk_dev.device.cmd_pipeline_barrier2(
                cmd, &pre_dep,
            );
        }

        self.scene_pipeline.record(
            &self.vk_dev.device,
            cmd,
            swapchain_view,
            extent,
            frame_idx,
            snapshot,
        );

        {
            let mut compositor = ActiveUiCompositor {
                vk_dev: &mut self.vk_dev,
                ui_pipeline: &mut self.ui_pipeline,
                cmd,
                swapchain_view,
                swapchain_extent: extent,
                frame_idx,
            };
            let mut ui_target = UiRenderTarget::new(
                extent.width,
                extent.height,
                self.window.scale_factor() as f32,
                &mut compositor,
            );
            ui.render(
                &mut ui_target,
                snapshot.ui_time_ms,
            );
        }

        let to_present = [vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
            .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::BOTTOM_OF_PIPE)
            .dst_access_mask(vk::AccessFlags2::empty())
            .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
            .image(swapchain_image)
            .subresource_range(color_subresource)];
        let post_dep = vk::DependencyInfo::default().image_memory_barriers(&to_present);

        // SAFETY: `cmd` is recording and all rendering passes have ended.
        unsafe {
            self.vk_dev.device.cmd_pipeline_barrier2(
                cmd, &post_dep,
            );
            if let Err(code) = self.vk_dev.device.end_command_buffer(cmd) {
                tracing::error!(
                    ?code,
                    "vkEndCommandBuffer failed"
                );
                return;
            }
        }

        let wait_infos = [vk::SemaphoreSubmitInfo::default()
            .semaphore(image_available)
            .stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)];
        let cmd_infos = [vk::CommandBufferSubmitInfo::default().command_buffer(cmd)];
        let signal_infos = [vk::SemaphoreSubmitInfo::default()
            .semaphore(render_finished)
            .stage_mask(vk::PipelineStageFlags2::ALL_GRAPHICS)];
        let submit_info = [vk::SubmitInfo2::default()
            .wait_semaphore_infos(&wait_infos)
            .command_buffer_infos(&cmd_infos)
            .signal_semaphore_infos(&signal_infos)];

        // SAFETY: Semaphores, command buffer, and fence are valid for this submit.
        if let Err(code) = unsafe {
            self.vk_dev.device.queue_submit2(
                self.vk_dev.graphics_queue,
                &submit_info,
                fence,
            )
        } {
            tracing::error!(
                ?code,
                "vkQueueSubmit2 failed"
            );
            return;
        }

        // Only register a compositor frame callback under VSync; on Wayland this throttles
        // `RedrawRequested` to the display refresh rate even with MAILBOX or IMMEDIATE.
        if self.frame_pacing == FramePacing::VSync {
            self.window.pre_present_notify();
        }

        let wait_semaphores = [render_finished];
        let swapchains = [self.swapchain.swapchain];
        let image_indices = [image_index];
        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(&wait_semaphores)
            .swapchains(&swapchains)
            .image_indices(&image_indices);

        // SAFETY: `image_index` was acquired and `render_finished` is signaled by the submit above.
        match unsafe {
            self.swapchain.swapchain_loader.queue_present(
                self.vk_dev.graphics_queue,
                &present_info,
            )
        } {
            Ok(present_suboptimal) if present_suboptimal || suboptimal => {
                let size = self.window.inner_size();
                if size.width > 0 && size.height > 0 {
                    self.resize(size);
                }
            }
            Ok(_) => {}
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                let size = self.window.inner_size();
                if size.width > 0 && size.height > 0 {
                    self.resize(size);
                }
            }
            Err(code) => {
                tracing::error!(
                    ?code,
                    "vkQueuePresentKHR failed"
                );
            }
        }

        self.frame_cursor = self.frame_cursor.wrapping_add(1);
    }
}

impl Drop for VulkanRenderer {
    fn drop(&mut self) {
        self.vk_dev.wait_idle();
        let device = self.vk_dev.device.clone();
        for frame in &self.frames {
            // SAFETY: Device is idle before teardown.
            unsafe {
                device.destroy_fence(
                    frame.in_flight,
                    None,
                );
                device.destroy_semaphore(
                    frame.image_available,
                    None,
                );
                device.destroy_command_pool(
                    frame.command_pool,
                    None,
                );
            }
        }
        for sem in self.render_finished.drain(..) {
            // SAFETY: Device is idle before teardown.
            unsafe {
                device.destroy_semaphore(
                    sem, None,
                );
            }
        }
        self.ui_pipeline.destroy(&mut self.vk_dev);
        self.scene_pipeline.destroy(&mut self.vk_dev);
        self.swapchain.destroy(&device);
    }
}

fn create_frame_sync(vk_dev: &VulkanDevice) -> Result<FrameSync, GpuError> {
    let pool_info = vk::CommandPoolCreateInfo::default()
        .queue_family_index(vk_dev.graphics_queue_family)
        .flags(vk::CommandPoolCreateFlags::TRANSIENT);
    // SAFETY: `vk_dev.device` is valid.
    let command_pool = unsafe {
        vk_dev
            .device
            .create_command_pool(
                &pool_info, None,
            )
            .map_err(
                |code| {
                    vk_err(
                        "create_command_pool",
                        code,
                    )
                },
            )?
    };

    let alloc_info = vk::CommandBufferAllocateInfo::default()
        .command_pool(command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);
    // SAFETY: `command_pool` was just created.
    let command_buffer = unsafe {
        vk_dev
            .device
            .allocate_command_buffers(&alloc_info)
            .map_err(
                |code| {
                    vk_err(
                        "allocate_command_buffers",
                        code,
                    )
                },
            )?[0]
    };

    let sem_info = vk::SemaphoreCreateInfo::default();
    // SAFETY: `vk_dev.device` is valid.
    let image_available = unsafe {
        vk_dev
            .device
            .create_semaphore(
                &sem_info, None,
            )
            .map_err(
                |code| {
                    vk_err(
                        "create_semaphore",
                        code,
                    )
                },
            )?
    };

    let fence_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
    // SAFETY: `vk_dev.device` is valid.
    let in_flight = unsafe {
        vk_dev
            .device
            .create_fence(
                &fence_info,
                None,
            )
            .map_err(
                |code| {
                    vk_err(
                        "create_fence",
                        code,
                    )
                },
            )?
    };

    Ok(
        FrameSync {
            command_pool,
            command_buffer,
            image_available,
            in_flight,
        },
    )
}

fn create_present_semaphores(
    device: &ash::Device,
    count: usize,
) -> Result<Vec<vk::Semaphore>, GpuError> {
    let mut sems = Vec::with_capacity(count);
    let sem_info = vk::SemaphoreCreateInfo::default();
    for _ in 0..count {
        // SAFETY: `device` is valid.
        match unsafe {
            device.create_semaphore(
                &sem_info, None,
            )
        } {
            Ok(sem) => sems.push(sem),
            Err(code) => {
                for sem in sems {
                    // SAFETY: Clean up created semaphores on error.
                    unsafe {
                        device.destroy_semaphore(
                            sem, None,
                        );
                    }
                }
                return Err(
                    vk_err(
                        "create_semaphore",
                        code,
                    ),
                );
            }
        }
    }
    Ok(sems)
}
