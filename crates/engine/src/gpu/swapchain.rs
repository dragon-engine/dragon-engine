#![allow(unsafe_code)]

use ash::vk;

use crate::gpu::device::VulkanDevice;
use crate::gpu::{GpuError, vk_err};
use crate::runner::FramePacing;

struct SwapchainBuildSpec {
    old_swapchain: vk::SwapchainKHR,
    width: u32,
    height: u32,
    frame_pacing: FramePacing,
}

struct BuiltSwapchain {
    swapchain: vk::SwapchainKHR,
    format: vk::Format,
    extent: vk::Extent2D,
    images: Vec<vk::Image>,
    image_views: Vec<vk::ImageView>,
}

/// Private Vulkan swapchain and surface owner.
/// Must be destroyed before `VulkanDevice` drops.
pub(crate) struct VulkanSwapchain {
    frame_pacing: FramePacing,
    surface_loader: ash::khr::surface::Instance,
    surface: vk::SurfaceKHR,
    pub(crate) swapchain_loader: ash::khr::swapchain::Device,
    pub(crate) swapchain: vk::SwapchainKHR,
    pub(crate) format: vk::Format,
    pub(crate) extent: vk::Extent2D,
    pub(crate) images: Vec<vk::Image>,
    pub(crate) image_views: Vec<vk::ImageView>,
}

impl VulkanSwapchain {
    pub(crate) fn new(
        vk_dev: &VulkanDevice,
        surface_loader: ash::khr::surface::Instance,
        surface: vk::SurfaceKHR,
        width: u32,
        height: u32,
        frame_pacing: FramePacing,
    ) -> Result<Self, GpuError> {
        let swapchain_loader = ash::khr::swapchain::Device::new(
            &vk_dev.instance,
            &vk_dev.device,
        );
        let built = build_swapchain(
            vk_dev,
            &surface_loader,
            surface,
            &swapchain_loader,
            SwapchainBuildSpec {
                old_swapchain: vk::SwapchainKHR::null(),
                width,
                height,
                frame_pacing,
            },
        )?;
        Ok(
            Self {
                frame_pacing,
                surface_loader,
                surface,
                swapchain_loader,
                swapchain: built.swapchain,
                format: built.format,
                extent: built.extent,
                images: built.images,
                image_views: built.image_views,
            },
        )
    }

    pub(crate) fn recreate(
        &mut self,
        vk_dev: &VulkanDevice,
        width: u32,
        height: u32,
    ) -> Result<(), GpuError> {
        vk_dev.wait_idle();
        let old_swapchain = self.swapchain;
        let built = build_swapchain(
            vk_dev,
            &self.surface_loader,
            self.surface,
            &self.swapchain_loader,
            SwapchainBuildSpec {
                old_swapchain,
                width,
                height,
                frame_pacing: self.frame_pacing,
            },
        )?;

        for view in self.image_views.drain(..) {
            // SAFETY: Device is idle and old views are replaced before returning.
            unsafe {
                vk_dev.device.destroy_image_view(
                    view, None,
                );
            }
        }
        if old_swapchain != vk::SwapchainKHR::null() {
            // SAFETY: Device is idle and new swapchain has taken ownership of the surface.
            unsafe {
                self.swapchain_loader.destroy_swapchain(
                    old_swapchain,
                    None,
                );
            }
        }

        self.swapchain = built.swapchain;
        self.format = built.format;
        self.extent = built.extent;
        self.images = built.images;
        self.image_views = built.image_views;
        Ok(())
    }

    pub(crate) fn destroy(&mut self, device: &ash::Device) {
        for view in self.image_views.drain(..) {
            // SAFETY: Caller waited for device idle before shutdown.
            unsafe {
                device.destroy_image_view(
                    view, None,
                );
            }
        }
        if self.swapchain != vk::SwapchainKHR::null() {
            // SAFETY: Caller waited for device idle and all views are destroyed.
            unsafe {
                self.swapchain_loader.destroy_swapchain(
                    self.swapchain,
                    None,
                );
            }
            self.swapchain = vk::SwapchainKHR::null();
        }
        if self.surface != vk::SurfaceKHR::null() {
            // SAFETY: Swapchain is destroyed and instance is still alive.
            unsafe {
                self.surface_loader.destroy_surface(
                    self.surface,
                    None,
                );
            }
            self.surface = vk::SurfaceKHR::null();
        }
    }
}

fn build_swapchain(
    vk_dev: &VulkanDevice,
    surface_loader: &ash::khr::surface::Instance,
    surface: vk::SurfaceKHR,
    swapchain_loader: &ash::khr::swapchain::Device,
    spec: SwapchainBuildSpec,
) -> Result<BuiltSwapchain, GpuError> {
    // SAFETY: `physical_device` and `surface` are valid.
    let caps = unsafe {
        surface_loader
            .get_physical_device_surface_capabilities(
                vk_dev.physical_device,
                surface,
            )
            .map_err(
                |code| {
                    vk_err(
                        "get_physical_device_surface_capabilities",
                        code,
                    )
                },
            )?
    };
    // SAFETY: `physical_device` and `surface` are valid.
    let formats = unsafe {
        surface_loader
            .get_physical_device_surface_formats(
                vk_dev.physical_device,
                surface,
            )
            .map_err(
                |code| {
                    vk_err(
                        "get_physical_device_surface_formats",
                        code,
                    )
                },
            )?
    };
    let surface_format = formats
        .iter()
        .copied()
        .find(
            |f| {
                matches!(
                    f.format,
                    vk::Format::B8G8R8A8_SRGB | vk::Format::R8G8B8A8_SRGB
                ) && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
            },
        )
        .or_else(|| formats.first().copied())
        .ok_or_else(
            || {
                GpuError::UnsupportedAdapter(
                    String::from("surface reported zero supported formats"),
                )
            },
        )?;

    // SAFETY: `physical_device` and `surface` are valid.
    let present_modes = unsafe {
        surface_loader
            .get_physical_device_surface_present_modes(
                vk_dev.physical_device,
                surface,
            )
            .map_err(
                |code| {
                    vk_err(
                        "get_physical_device_surface_present_modes",
                        code,
                    )
                },
            )?
    };
    let present_mode = select_present_mode(
        &present_modes,
        spec.frame_pacing,
    );

    let extent = if caps.current_extent.width != u32::MAX && caps.current_extent.height != u32::MAX
    {
        vk::Extent2D {
            width: caps.current_extent.width.max(1),
            height: caps.current_extent.height.max(1),
        }
    } else {
        vk::Extent2D {
            width: spec.width.max(1).clamp(
                caps.min_image_extent.width.max(1),
                caps.max_image_extent.width.max(1),
            ),
            height: spec.height.max(1).clamp(
                caps.min_image_extent.height.max(1),
                caps.max_image_extent.height.max(1),
            ),
        }
    };

    let mut image_count = caps.min_image_count.saturating_add(1).max(2);
    if caps.max_image_count > 0 && image_count > caps.max_image_count {
        image_count = caps.max_image_count;
    }

    let create_info = vk::SwapchainCreateInfoKHR::default()
        .surface(surface)
        .min_image_count(image_count)
        .image_format(surface_format.format)
        .image_color_space(surface_format.color_space)
        .image_extent(extent)
        .image_array_layers(1)
        .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
        .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
        .pre_transform(caps.current_transform)
        .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
        .present_mode(present_mode)
        .clipped(true)
        .old_swapchain(spec.old_swapchain);

    // SAFETY: `create_info` uses capabilities and formats queried from this surface/device.
    let swapchain = unsafe {
        swapchain_loader
            .create_swapchain(
                &create_info,
                None,
            )
            .map_err(
                |code| {
                    vk_err(
                        "create_swapchain",
                        code,
                    )
                },
            )?
    };

    // SAFETY: `swapchain` was just created.
    let images = unsafe {
        swapchain_loader.get_swapchain_images(swapchain).map_err(
            |code| {
                vk_err(
                    "get_swapchain_images",
                    code,
                )
            },
        )?
    };

    let mut image_views = Vec::with_capacity(images.len());
    for &image in &images {
        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(surface_format.format)
            .subresource_range(
                vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                },
            );
        // SAFETY: `image` belongs to `swapchain` with `surface_format.format`.
        match unsafe {
            vk_dev.device.create_image_view(
                &view_info, None,
            )
        } {
            Ok(view) => image_views.push(view),
            Err(code) => {
                for v in image_views {
                    // SAFETY: Clean up created views on error.
                    unsafe {
                        vk_dev.device.destroy_image_view(
                            v, None,
                        );
                    }
                }
                // SAFETY: Clean up newly created swapchain on error.
                unsafe {
                    swapchain_loader.destroy_swapchain(
                        swapchain, None,
                    );
                }
                return Err(
                    vk_err(
                        "create_image_view",
                        code,
                    ),
                );
            }
        }
    }

    Ok(
        BuiltSwapchain {
            swapchain,
            format: surface_format.format,
            extent,
            images,
            image_views,
        },
    )
}

fn select_present_mode(
    present_modes: &[vk::PresentModeKHR],
    frame_pacing: FramePacing,
) -> vk::PresentModeKHR {
    match frame_pacing {
        FramePacing::VSync => vk::PresentModeKHR::FIFO,
        FramePacing::Capped(_) | FramePacing::Unlimited => {
            if present_modes.contains(&vk::PresentModeKHR::MAILBOX) {
                vk::PresentModeKHR::MAILBOX
            } else if present_modes.contains(&vk::PresentModeKHR::IMMEDIATE) {
                vk::PresentModeKHR::IMMEDIATE
            } else {
                vk::PresentModeKHR::FIFO
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vsync_always_selects_fifo() {
        let modes = [
            vk::PresentModeKHR::IMMEDIATE,
            vk::PresentModeKHR::MAILBOX,
            vk::PresentModeKHR::FIFO,
        ];
        assert_eq!(
            select_present_mode(
                &modes,
                FramePacing::VSync,
            ),
            vk::PresentModeKHR::FIFO
        );
    }

    #[test]
    fn unlimited_and_capped_prefer_mailbox_then_immediate_then_fifo() {
        let with_mailbox = [
            vk::PresentModeKHR::FIFO,
            vk::PresentModeKHR::IMMEDIATE,
            vk::PresentModeKHR::MAILBOX,
        ];
        let without_mailbox = [vk::PresentModeKHR::FIFO, vk::PresentModeKHR::IMMEDIATE];
        let fifo_only = [vk::PresentModeKHR::FIFO];

        for pacing in [FramePacing::Unlimited, FramePacing::Capped(144)] {
            assert_eq!(
                select_present_mode(
                    &with_mailbox,
                    pacing,
                ),
                vk::PresentModeKHR::MAILBOX
            );
            assert_eq!(
                select_present_mode(
                    &without_mailbox,
                    pacing,
                ),
                vk::PresentModeKHR::IMMEDIATE
            );
            assert_eq!(
                select_present_mode(
                    &fifo_only, pacing,
                ),
                vk::PresentModeKHR::FIFO
            );
        }
    }
}
