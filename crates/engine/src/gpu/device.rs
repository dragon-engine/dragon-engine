#![allow(unsafe_code)]

use std::ffi::{CStr, CString};

use ash::vk;
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};
use gpu_allocator::{AllocationSizes, AllocatorDebugSettings};
use winit::raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use winit::window::Window;

use crate::gpu::{GpuError, decode_spirv_words, vk_err};

/// Runtime Vulkan 1.3 capabilities probed from the selected physical device.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub(crate) struct VulkanCapabilities {
    pub(crate) api_version: u32,
    pub(crate) device_name: String,
    pub(crate) device_type: vk::PhysicalDeviceType,
    pub(crate) min_uniform_buffer_offset_alignment: u64,
    pub(crate) ray_query: bool,
    pub(crate) mesh_shader: bool,
    pub(crate) descriptor_indexing_update_after_bind: bool,
    pub(crate) runtime_descriptor_array: bool,
    pub(crate) storage_image_extended_formats: bool,
    pub(crate) buffer_device_address: bool,
}

/// Private Vulkan 1.3 instance, logical device, graphics queue, and `gpu-allocator` owner.
pub(crate) struct VulkanDevice {
    _entry: ash::Entry,
    pub(crate) instance: ash::Instance,
    pub(crate) physical_device: vk::PhysicalDevice,
    pub(crate) device: ash::Device,
    pub(crate) graphics_queue: vk::Queue,
    pub(crate) graphics_queue_family: u32,
    allocator: Option<Allocator>,
    pub(crate) capabilities: VulkanCapabilities,
}

impl VulkanDevice {
    pub(crate) fn new_windowed(
        window: &Window,
    ) -> Result<
        (
            Self,
            ash::khr::surface::Instance,
            vk::SurfaceKHR,
        ),
        GpuError,
    > {
        let display_handle = window
            .display_handle()
            .map_err(|err| GpuError::WindowHandle(format!("display handle unavailable: {err}")))?;
        let window_handle = window
            .window_handle()
            .map_err(|err| GpuError::WindowHandle(format!("window handle unavailable: {err}")))?;

        let surface_exts = ash_window::enumerate_required_extensions(display_handle.as_raw())
            .map_err(
                |code| {
                    vk_err(
                        "enumerate_required_extensions",
                        code,
                    )
                },
            )?;

        // SAFETY: Loading the platform Vulkan loader (`libvulkan.so.1` / `vulkan-1.dll`).
        let entry = unsafe { ash::Entry::load().map_err(|err| GpuError::Loader(err.to_string()))? };
        let instance = create_instance(
            &entry,
            surface_exts,
        )?;
        let surface_loader = ash::khr::surface::Instance::new(
            &entry, &instance,
        );

        // SAFETY: `entry`, `instance`, and raw window/display handles are valid for this call.
        let surface = unsafe {
            ash_window::create_surface(
                &entry,
                &instance,
                display_handle.as_raw(),
                window_handle.as_raw(),
                None,
            )
            .map_err(
                |code| {
                    vk_err(
                        "create_surface",
                        code,
                    )
                },
            )?
        };

        let vk_dev = match create_device_and_allocator(
            entry,
            instance,
            Some(
                (
                    &surface_loader,
                    surface,
                ),
            ),
        ) {
            Ok(dev) => dev,
            Err(err) => {
                // SAFETY: Surface creation succeeded; destroy it before returning the error.
                // Note: `instance` was moved into `create_device_and_allocator` and destroyed there on error.
                // To guarantee surface is destroyed BEFORE instance, `create_device_and_allocator` destroys surface on error if needed.
                return Err(err);
            }
        };

        Ok(
            (
                vk_dev,
                surface_loader,
                surface,
            ),
        )
    }

    pub(crate) fn new_headless() -> Result<Self, GpuError> {
        // SAFETY: Loading the platform Vulkan loader (`libvulkan.so.1` / `vulkan-1.dll`).
        let entry = unsafe { ash::Entry::load().map_err(|err| GpuError::Loader(err.to_string()))? };
        let instance = create_instance(
            &entry,
            &[],
        )?;
        create_device_and_allocator(
            entry, instance, None,
        )
    }

    pub(crate) fn allocator_mut(&mut self) -> &mut Allocator {
        self.allocator
            .as_mut()
            .expect("gpu-allocator alive for lifetime of VulkanDevice")
    }

    pub(crate) fn create_shader_module(
        &self,
        spirv_bytes: &[u8],
    ) -> Result<vk::ShaderModule, GpuError> {
        let words = decode_spirv_words(spirv_bytes)?;
        let info = vk::ShaderModuleCreateInfo::default().code(&words);
        // SAFETY: `words` is a validated 4-byte aligned SPIR-V binary and `self.device` is live.
        unsafe {
            self.device
                .create_shader_module(
                    &info, None,
                )
                .map_err(
                    |code| {
                        vk_err(
                            "create_shader_module",
                            code,
                        )
                    },
                )
        }
    }

    pub(crate) fn wait_idle(&self) {
        // SAFETY: `self.device` is valid and host-synchronized on shutdown/resize.
        if let Err(code) = unsafe { self.device.device_wait_idle() } {
            tracing::error!(
                ?code,
                "vkDeviceWaitIdle failed"
            );
        }
    }
}

impl Drop for VulkanDevice {
    fn drop(&mut self) {
        self.wait_idle();
        // Drop `gpu-allocator` BEFORE destroying the Vulkan logical device so every `VkDeviceMemory`
        // block is freed cleanly with zero validation errors.
        self.allocator = None;
        // SAFETY: All child pipelines, buffers, images, swapchains, and allocations have been destroyed.
        unsafe {
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

fn create_instance(
    entry: &ash::Entry,
    extra_extensions: &[*const i8],
) -> Result<ash::Instance, GpuError> {
    let app_name = c"Dragon Engine";
    let engine_name = c"Dragon Engine";
    let app_info = vk::ApplicationInfo::default()
        .application_name(app_name)
        .application_version(
            vk::make_api_version(
                0, 0, 1, 0,
            ),
        )
        .engine_name(engine_name)
        .engine_version(
            vk::make_api_version(
                0, 0, 1, 0,
            ),
        )
        .api_version(vk::API_VERSION_1_3);

    let mut enabled_layers: Vec<*const i8> = Vec::new();
    let validation_layer = c"VK_LAYER_KHRONOS_validation";
    if cfg!(debug_assertions) {
        // SAFETY: Querying static instance layer properties from the Vulkan loader.
        if let Ok(available_layers) = unsafe { entry.enumerate_instance_layer_properties() } {
            let has_validation = available_layers.iter().any(
                |layer| {
                    // SAFETY: `layer_name` is a null-terminated Vulkan C string.
                    let name = unsafe { CStr::from_ptr(layer.layer_name.as_ptr()) };
                    name == validation_layer
                },
            );
            if has_validation {
                enabled_layers.push(validation_layer.as_ptr());
            }
        }
    }

    let create_info = vk::InstanceCreateInfo::default()
        .application_info(&app_info)
        .enabled_layer_names(&enabled_layers)
        .enabled_extension_names(extra_extensions);

    // SAFETY: All pointer slices remain valid for the duration of `create_instance`.
    unsafe {
        entry
            .create_instance(
                &create_info,
                None,
            )
            .map_err(
                |code| {
                    vk_err(
                        "create_instance",
                        code,
                    )
                },
            )
    }
}

fn create_device_and_allocator(
    entry: ash::Entry,
    instance: ash::Instance,
    surface: Option<(
        &ash::khr::surface::Instance,
        vk::SurfaceKHR,
    )>,
) -> Result<VulkanDevice, GpuError> {
    let destroy_on_error = |inst: &ash::Instance| {
        if let Some((surface_loader, surf)) = surface {
            // SAFETY: Destroy surface before instance on initialization failure.
            unsafe {
                surface_loader.destroy_surface(
                    surf, None,
                );
            }
        }
        // SAFETY: Destroy instance on initialization failure.
        unsafe {
            inst.destroy_instance(None);
        }
    };

    // SAFETY: `instance` is valid.
    let physical_devices = match unsafe { instance.enumerate_physical_devices() } {
        Ok(devs) if !devs.is_empty() => devs,
        Ok(_) => {
            destroy_on_error(&instance);
            return Err(
                GpuError::UnsupportedAdapter(String::from("no Vulkan physical devices found")),
            );
        }
        Err(code) => {
            destroy_on_error(&instance);
            return Err(
                vk_err(
                    "enumerate_physical_devices",
                    code,
                ),
            );
        }
    };

    let Some((physical_device, queue_family, capabilities, has_swapchain)) = select_physical_device(
        &instance,
        &physical_devices,
        surface,
    ) else {
        destroy_on_error(&instance);
        return Err(
            GpuError::UnsupportedAdapter(
                String::from(
                    "no Vulkan 1.3 physical device supporting dynamic_rendering and synchronization2 found",
                ),
            ),
        );
    };

    let mut device_extensions: Vec<*const i8> = Vec::new();
    if surface.is_some() {
        if !has_swapchain {
            destroy_on_error(&instance);
            return Err(
                GpuError::UnsupportedAdapter(
                    String::from("selected adapter lacks VK_KHR_swapchain"),
                ),
            );
        }
        device_extensions.push(ash::khr::swapchain::NAME.as_ptr());
    }

    let queue_priorities = [1.0_f32];
    let queue_infos = [vk::DeviceQueueCreateInfo::default()
        .queue_family_index(queue_family)
        .queue_priorities(&queue_priorities)];

    let mut features12 = vk::PhysicalDeviceVulkan12Features::default()
        .timeline_semaphore(true)
        .buffer_device_address(capabilities.buffer_device_address);
    let mut features13 = vk::PhysicalDeviceVulkan13Features::default()
        .dynamic_rendering(true)
        .synchronization2(true);
    let mut features2 = vk::PhysicalDeviceFeatures2::default()
        .push_next(&mut features12)
        .push_next(&mut features13);

    let device_create_info = vk::DeviceCreateInfo::default()
        .queue_create_infos(&queue_infos)
        .enabled_extension_names(&device_extensions)
        .push_next(&mut features2);

    // SAFETY: `physical_device` was enumerated from `instance` and requested features were verified.
    let device = match unsafe {
        instance.create_device(
            physical_device,
            &device_create_info,
            None,
        )
    } {
        Ok(dev) => dev,
        Err(code) => {
            destroy_on_error(&instance);
            return Err(
                vk_err(
                    "create_device",
                    code,
                ),
            );
        }
    };

    // SAFETY: `queue_family` was requested in `queue_infos` with count 1.
    let graphics_queue = unsafe {
        device.get_device_queue(
            queue_family,
            0,
        )
    };

    let allocator = match Allocator::new(
        &AllocatorCreateDesc {
            instance: instance.clone(),
            device: device.clone(),
            physical_device,
            debug_settings: AllocatorDebugSettings::default(),
            buffer_device_address: capabilities.buffer_device_address,
            allocation_sizes: AllocationSizes::default(),
        },
    ) {
        Ok(alloc) => alloc,
        Err(err) => {
            // SAFETY: Clean up logical device, surface, and instance on allocator creation failure.
            unsafe {
                device.destroy_device(None);
            }
            destroy_on_error(&instance);
            return Err(GpuError::Allocator(err.to_string()));
        }
    };

    tracing::info!(
        device = %capabilities.device_name,
        api_major = vk::api_version_major(capabilities.api_version),
        api_minor = vk::api_version_minor(capabilities.api_version),
        ray_query = capabilities.ray_query,
        mesh_shader = capabilities.mesh_shader,
        "Initialized Vulkan 1.3 device"
    );

    Ok(
        VulkanDevice {
            _entry: entry,
            instance,
            physical_device,
            device,
            graphics_queue,
            graphics_queue_family: queue_family,
            allocator: Some(allocator),
            capabilities,
        },
    )
}

fn select_physical_device(
    instance: &ash::Instance,
    devices: &[vk::PhysicalDevice],
    surface: Option<(
        &ash::khr::surface::Instance,
        vk::SurfaceKHR,
    )>,
) -> Option<(
    vk::PhysicalDevice,
    u32,
    VulkanCapabilities,
    bool,
)> {
    let mut best: Option<(
        u32,
        vk::PhysicalDevice,
        u32,
        VulkanCapabilities,
        bool,
    )> = None;

    for &pdev in devices {
        // SAFETY: `pdev` was enumerated from `instance`.
        let props = unsafe { instance.get_physical_device_properties(pdev) };
        if props.api_version < vk::API_VERSION_1_3 {
            continue;
        }

        // SAFETY: `pdev` is a valid Vulkan 1.3 physical device.
        let mut features12 = vk::PhysicalDeviceVulkan12Features::default();
        let mut features13 = vk::PhysicalDeviceVulkan13Features::default();
        let core_features = {
            let mut features2 = vk::PhysicalDeviceFeatures2::default()
                .push_next(&mut features12)
                .push_next(&mut features13);
            unsafe {
                instance.get_physical_device_features2(
                    pdev,
                    &mut features2,
                );
            }
            features2.features
        };

        if features13.dynamic_rendering == vk::FALSE
            || features13.synchronization2 == vk::FALSE
            || features12.timeline_semaphore == vk::FALSE
        {
            continue;
        }

        // SAFETY: `pdev` is valid.
        let queue_families = unsafe { instance.get_physical_device_queue_family_properties(pdev) };
        let Some(queue_family) = queue_families.iter().enumerate().find_map(
            |(idx, info)| {
                let idx_u32 = u32::try_from(idx).ok()?;
                if !info.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                    return None;
                }
                if let Some((surface_loader, surf)) = surface {
                    // SAFETY: `pdev`, `idx_u32`, and `surf` are valid.
                    let present_ok = unsafe {
                        surface_loader
                            .get_physical_device_surface_support(
                                pdev, idx_u32, surf,
                            )
                            .unwrap_or(false)
                    };
                    if !present_ok {
                        return None;
                    }
                }
                Some(idx_u32)
            },
        ) else {
            continue;
        };

        // SAFETY: `pdev` is valid.
        let ext_props = unsafe {
            instance
                .enumerate_device_extension_properties(pdev)
                .unwrap_or_default()
        };
        let has_ext = |target: &CStr| {
            ext_props.iter().any(
                |ext| {
                    // SAFETY: `extension_name` is a null-terminated Vulkan C string.
                    let name = unsafe { CStr::from_ptr(ext.extension_name.as_ptr()) };
                    name == target
                },
            )
        };

        let has_swapchain = has_ext(ash::khr::swapchain::NAME);
        if surface.is_some() && !has_swapchain {
            continue;
        }

        let ray_query = has_ext(vk::KHR_RAY_QUERY_NAME)
            && has_ext(vk::KHR_ACCELERATION_STRUCTURE_NAME)
            && has_ext(vk::KHR_DEFERRED_HOST_OPERATIONS_NAME);
        let mesh_shader = has_ext(vk::EXT_MESH_SHADER_NAME);

        // SAFETY: `device_name` is a null-terminated Vulkan C string.
        let device_name = unsafe { CStr::from_ptr(props.device_name.as_ptr()) }
            .to_string_lossy()
            .into_owned();

        let caps = VulkanCapabilities {
            api_version: props.api_version,
            device_name,
            device_type: props.device_type,
            min_uniform_buffer_offset_alignment: props.limits.min_uniform_buffer_offset_alignment,
            ray_query,
            mesh_shader,
            descriptor_indexing_update_after_bind: features12
                .descriptor_binding_sampled_image_update_after_bind
                == vk::TRUE,
            runtime_descriptor_array: features12.runtime_descriptor_array == vk::TRUE,
            storage_image_extended_formats: core_features.shader_storage_image_extended_formats
                == vk::TRUE,
            buffer_device_address: features12.buffer_device_address == vk::TRUE,
        };

        let mut score = match props.device_type {
            vk::PhysicalDeviceType::DISCRETE_GPU => 1000,
            vk::PhysicalDeviceType::INTEGRATED_GPU => 500,
            vk::PhysicalDeviceType::VIRTUAL_GPU => 200,
            _ => 50,
        };
        if ray_query {
            score += 100;
        }
        if mesh_shader {
            score += 50;
        }

        if best
            .as_ref()
            .is_none_or(|(best_score, ..)| score > *best_score)
        {
            best = Some(
                (
                    score,
                    pdev,
                    queue_family,
                    caps,
                    has_swapchain,
                ),
            );
        }
    }

    best.map(
        |(_, pdev, qf, caps, has_swapchain)| {
            (
                pdev,
                qf,
                caps,
                has_swapchain,
            )
        },
    )
}

pub(crate) fn entry_name_cstring(name: &str) -> Result<CString, GpuError> {
    CString::new(name).map_err(
        |err| {
            GpuError::InvalidSpirv(
                format!("shader entry point `{name}` contains interior NUL: {err}"),
            )
        },
    )
}
