#![allow(unsafe_code)]

pub(crate) mod device;
pub(crate) mod renderer;
pub(crate) mod scene_pipeline;
pub(crate) mod swapchain;
pub(crate) mod ui_pipeline;

use std::fmt;

use ash::vk;
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator};

const SPIRV_MAGIC: u32 = 0x0723_0203;

/// Typed error returned by the private Vulkan 1.3 backend.
#[derive(Debug)]
pub enum GpuError {
    Loader(String),
    Vulkan { op: &'static str, code: vk::Result },
    Allocator(String),
    UnsupportedAdapter(String),
    InvalidSpirv(String),
    WindowHandle(String),
}

impl fmt::Display for GpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Loader(msg) => write!(
                f,
                "Vulkan loader error: {msg}"
            ),
            Self::Vulkan { op, code } => write!(
                f,
                "Vulkan call `{op}` failed: {code:?}"
            ),
            Self::Allocator(msg) => write!(
                f,
                "GPU memory allocation failed: {msg}"
            ),
            Self::UnsupportedAdapter(msg) => write!(
                f,
                "Unsupported Vulkan adapter: {msg}"
            ),
            Self::InvalidSpirv(msg) => write!(
                f,
                "Invalid SPIR-V shader module: {msg}"
            ),
            Self::WindowHandle(msg) => write!(
                f,
                "Window handle error: {msg}"
            ),
        }
    }
}

impl std::error::Error for GpuError {}

pub(crate) fn vk_err(op: &'static str, code: vk::Result) -> GpuError {
    GpuError::Vulkan { op, code }
}

/// Validates SPIR-V header magic and decodes little-endian `u32` words safely without alignment UB.
pub(crate) fn decode_spirv_words(bytes: &[u8]) -> Result<Vec<u32>, GpuError> {
    if bytes.len() < 20 || !bytes.len().is_multiple_of(4) {
        return Err(
            GpuError::InvalidSpirv(
                format!(
                    "expected non-empty 4-byte aligned SPIR-V payload, got {} bytes",
                    bytes.len()
                ),
            ),
        );
    }
    let (chunks, _) = bytes.as_chunks::<4>();
    let words: Vec<u32> = chunks.iter().copied().map(u32::from_le_bytes).collect();
    if words[0] != SPIRV_MAGIC {
        return Err(
            GpuError::InvalidSpirv(
                format!(
                    "unexpected SPIR-V magic 0x{:08x} (expected 0x{SPIRV_MAGIC:08x})",
                    words[0]
                ),
            ),
        );
    }
    Ok(words)
}

/// Owned Vulkan buffer paired with its `gpu-allocator` allocation.
pub(crate) struct GpuBuffer {
    pub(crate) buffer: vk::Buffer,
    pub(crate) allocation: Option<Allocation>,
    pub(crate) size: u64,
}

impl GpuBuffer {
    pub(crate) fn create(
        device: &ash::Device,
        allocator: &mut Allocator,
        name: &str,
        size: u64,
        usage: vk::BufferUsageFlags,
        location: MemoryLocation,
    ) -> Result<Self, GpuError> {
        let size = size.max(16);
        let create_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        // SAFETY: `device` is valid and `create_info` has valid size and usage flags.
        let buffer = unsafe {
            device
                .create_buffer(
                    &create_info,
                    None,
                )
                .map_err(
                    |code| {
                        vk_err(
                            "create_buffer",
                            code,
                        )
                    },
                )?
        };

        // SAFETY: `buffer` was just created by `device`.
        let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
        let allocation = allocator
            .allocate(
                &AllocationCreateDesc {
                    name,
                    requirements,
                    location,
                    linear: true,
                    allocation_scheme: AllocationScheme::GpuAllocatorManaged,
                },
            )
            .map_err(
                |err| {
                    // SAFETY: Clean up the buffer handle if memory allocation fails.
                    unsafe {
                        device.destroy_buffer(
                            buffer, None,
                        );
                    }
                    GpuError::Allocator(format!("{name}: {err}"))
                },
            )?;

        // SAFETY: `allocation` satisfies `requirements` for `buffer` and has not been bound.
        if let Err(code) = unsafe {
            device.bind_buffer_memory(
                buffer,
                allocation.memory(),
                allocation.offset(),
            )
        } {
            if let Err(free_err) = allocator.free(allocation) {
                tracing::error!(
                    ?free_err,
                    name,
                    "Failed to free GPU buffer memory after bind error"
                );
            }
            // SAFETY: Buffer is unbound and unused.
            unsafe {
                device.destroy_buffer(
                    buffer, None,
                );
            }
            return Err(
                vk_err(
                    "bind_buffer_memory",
                    code,
                ),
            );
        }

        Ok(
            Self {
                buffer,
                allocation: Some(allocation),
                size,
            },
        )
    }

    pub(crate) fn write_bytes(&mut self, offset: usize, bytes: &[u8]) -> Result<(), GpuError> {
        let Some(allocation) = self.allocation.as_mut() else {
            return Err(GpuError::Allocator(String::from("buffer allocation already freed")));
        };
        let Some(mapped) = allocation.mapped_slice_mut() else {
            return Err(GpuError::Allocator(String::from("buffer is not host-mapped")));
        };
        let end = offset.saturating_add(bytes.len());
        if end > mapped.len() {
            return Err(
                GpuError::Allocator(
                    format!(
                        "buffer write out of bounds: {end} > {}",
                        mapped.len()
                    ),
                ),
            );
        }
        mapped[offset..end].copy_from_slice(bytes);
        Ok(())
    }

    pub(crate) fn destroy(&mut self, device: &ash::Device, allocator: &mut Allocator) {
        if let Some(allocation) = self.allocation.take()
            && let Err(err) = allocator.free(allocation)
        {
            tracing::error!(
                ?err,
                "Failed to free GPU buffer allocation"
            );
        }
        if self.buffer != vk::Buffer::null() {
            // SAFETY: Caller guarantees GPU work referencing this buffer has completed.
            unsafe {
                device.destroy_buffer(
                    self.buffer,
                    None,
                );
            }
            self.buffer = vk::Buffer::null();
        }
    }
}

/// Owned Vulkan 2D image paired with its view and `gpu-allocator` allocation.
pub(crate) struct GpuImage {
    pub(crate) image: vk::Image,
    pub(crate) view: vk::ImageView,
    pub(crate) allocation: Option<Allocation>,
    pub(crate) width: u32,
    pub(crate) height: u32,
}

impl GpuImage {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn create_2d(
        device: &ash::Device,
        allocator: &mut Allocator,
        name: &str,
        width: u32,
        height: u32,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
        aspect: vk::ImageAspectFlags,
    ) -> Result<Self, GpuError> {
        let width = width.max(1);
        let height = height.max(1);
        let create_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(
                vk::Extent3D {
                    width,
                    height,
                    depth: 1,
                },
            )
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);

        // SAFETY: `device` is valid and `create_info` specifies a valid 2D image.
        let image = unsafe {
            device
                .create_image(
                    &create_info,
                    None,
                )
                .map_err(
                    |code| {
                        vk_err(
                            "create_image",
                            code,
                        )
                    },
                )?
        };

        // SAFETY: `image` was created by `device`.
        let requirements = unsafe { device.get_image_memory_requirements(image) };
        let allocation = allocator
            .allocate(
                &AllocationCreateDesc {
                    name,
                    requirements,
                    location: MemoryLocation::GpuOnly,
                    linear: false,
                    allocation_scheme: AllocationScheme::GpuAllocatorManaged,
                },
            )
            .map_err(
                |err| {
                    // SAFETY: Clean up unbound image on allocation failure.
                    unsafe {
                        device.destroy_image(
                            image, None,
                        );
                    }
                    GpuError::Allocator(format!("{name}: {err}"))
                },
            )?;

        // SAFETY: `allocation` satisfies `requirements` and has not been bound.
        if let Err(code) = unsafe {
            device.bind_image_memory(
                image,
                allocation.memory(),
                allocation.offset(),
            )
        } {
            if let Err(free_err) = allocator.free(allocation) {
                tracing::error!(
                    ?free_err,
                    name,
                    "Failed to free GPU image memory after bind error"
                );
            }
            // SAFETY: Image is unbound and unused.
            unsafe {
                device.destroy_image(
                    image, None,
                );
            }
            return Err(
                vk_err(
                    "bind_image_memory",
                    code,
                ),
            );
        }

        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .subresource_range(
                vk::ImageSubresourceRange {
                    aspect_mask: aspect,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                },
            );
        // SAFETY: `image` is valid and bound to device memory.
        let view = match unsafe {
            device.create_image_view(
                &view_info, None,
            )
        } {
            Ok(view) => view,
            Err(code) => {
                if let Err(free_err) = allocator.free(allocation) {
                    tracing::error!(
                        ?free_err,
                        name,
                        "Failed to free GPU image memory after view error"
                    );
                }
                // SAFETY: Image is unused after view creation error.
                unsafe {
                    device.destroy_image(
                        image, None,
                    );
                }
                return Err(
                    vk_err(
                        "create_image_view",
                        code,
                    ),
                );
            }
        };

        Ok(
            Self {
                image,
                view,
                allocation: Some(allocation),
                width,
                height,
            },
        )
    }

    pub(crate) fn destroy(&mut self, device: &ash::Device, allocator: &mut Allocator) {
        if self.view != vk::ImageView::null() {
            // SAFETY: Caller guarantees GPU work referencing this view has completed.
            unsafe {
                device.destroy_image_view(
                    self.view, None,
                );
            }
            self.view = vk::ImageView::null();
        }
        if self.image != vk::Image::null() {
            // SAFETY: Caller guarantees GPU work referencing this image has completed.
            unsafe {
                device.destroy_image(
                    self.image, None,
                );
            }
            self.image = vk::Image::null();
        }
        if let Some(allocation) = self.allocation.take()
            && let Err(err) = allocator.free(allocation)
        {
            tracing::error!(
                ?err,
                "Failed to free GPU image allocation"
            );
        }
    }
}
