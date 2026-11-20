# Implementation Plan: Dual-Camera 3D RPG Graphics & Streaming Architecture

This document defines the phased implementation plan for realizing [Architectural Design Specifications for a Dual-Camera 3D RPG Graphics Engine](graphics-architecture.md) inside Dragon Engine while preserving all runtime invariants in [Engine Architecture Principles](engine-architecture.md).

---

## 1. Locked Architectural Decisions

These decisions are final and must not be reopened during implementation without an explicit Blueprint review:

1. **Vulkan 1.4 Only via Private `ash 0.38` + `gpu-allocator 0.28` (Upgrade to `ash 0.39` on Release):** Target native **Vulkan 1.4** exclusively (Linux including Steam Deck, and Windows) through private `ash 0.38`, `ash-window 0.13`, and pure-Rust `gpu-allocator 0.28` inside `crates/engine/src/gpu/`. Startup requires a conformant Vulkan 1.4 adapter (`api_version >= vk::make_api_version(0, 1, 4, 0)`) and invokes promoted Vulkan 1.4 features (`push_descriptor`, `maintenance5`, `maintenance6`, `dynamic_rendering_local_read`, optional `host_image_copy`) through `ash 0.38`'s KHR/EXT loaders until `ash 0.39` + matching `gpu-allocator` release on crates.io. Native macOS, DX12, Metal, WebAssembly, and `wgpu` are out of scope. Zero raw Vulkan handles are exposed outside `crates/engine`.
2. **Strict 100% `rust-gpu` Vulkan 1.4 SPIR-V with Multi-Module Isolation (Zero WGSL, Zero Shader Cross-Compilers):** All engine-authored and game-authored compute, mesh, ray-query, graphics, upscaling (`TaauRcas`), and declarative `RenderEffect` shaders live in `crates/engine/shaders/core_shader` and external `rust-gpu` crates (such as `crates/fx_fresnel_rim`) compiled with `spirv-builder 0.11.0` (`spirv-unknown-vulkan1.4`, `SpirvBuilder::multimodule(true)` so optional `TaskEXT`/`MeshEXT` modules stay isolated from non-mesh GPUs, with a separate `core_shader_mesh` crate as explicit Phase 0 fallback) into Cargo `OUT_DIR` and loaded directly via `vkCreateShaderModule`. Shaders standardize on bound storage buffers (Set 1 `pushDescriptor`) + pool discriminators + push-constant element offsets (no shader-side `PhysicalStorageBuffer` pointers, keeping host-side BDA strictly for Vulkan AS builds). All runtime and `crates/asset_pipeline` crates use strictly pure-Rust dependencies (with `spirv-builder`'s transitive `spirv-tools-sys` as the sole shader-build exception).
3. **Scoped `unsafe` Isolation & Real-HW Validation:** Keep `unsafe_code = "deny"` in root `Cargo.toml` across all public APIs, simulation, gameplay, `shader_abi`, `asset_pipeline`, and `fx_fresnel_rim`. Isolate `#[allow(unsafe_code)]` strictly inside two private audited boundaries with explicit `// SAFETY:` comments and mandatory `VK_LAYER_KHRONOS_validation` testing directly on real Vulkan 1.4 ray-tracing hardware:
   - `crates/engine/src/gpu/` (private Vulkan 1.4 `ash` + `gpu-allocator` backend), and
   - `crates/engine/src/assets/mmap_store.rs` (read-only immutable `mmap` slices).
4. **Zero-Loading-Screen Async Pop-In (Frame-Pacing First):** Never stall simulation or rendering on disk I/O, decompression, GPU upload, BLAS build/compaction, or physics splicing. Accept visible asynchronous pop-in over frame-rate stutter.
5. **Independent Per-Subsystem Residency Policies & Dynamic VRAM Clamping:** Physics, Geometry/LOD, Ray-Tracing BLAS, Actors/AI, and GPU-Feedback Mipmap Streaming each own independent spatial residency policies and scaling parameters (no coupled monolithic sector rings), with VRAM pools dynamically clamped via `VK_EXT_memory_budget` at startup.
6. **Lightweight Streaming Pacing (No Runtime Millisecond Timers):** Use only a fixed-size Pass 1 staging ring (`STAGING_RING_MB`, optionally supplemented by Vulkan 1.4 `hostImageCopy` for small mip tails and UI RGBA8), a runtime-preset per-frame static BLAS build cap (`STATIC_BLAS_BUILDS_PER_FRAME`), and `VK_EXT_memory_budget`-clamped VRAM budgets (`STATIC_GEOMETRY_ARENA_MB`, `BLAS_POOL_BUDGET_MB`, `TEXTURE_STREAM_BUDGET_MB`). Millisecond figures are design-time references and offline QA profiler targets, never runtime timers.
7. **Zero-Lag Pass 2 / Pass 3 Ordering & 11-Pass Unified Viewmodel Pipeline:** Execute **Pass 2 (Compute Meshlet Skinning + Dynamic TOD Sky LUT / Amortized IBL Compute)** before **Pass 3 (In-Place Dynamic BLAS Refit, Async Static BLAS Compaction, and Active-Radius TLAS Rebuild)** and **Pass 7 (Demodulated ReSTIR DI, Reflections, Radiance Cascades)** so rasterization, ray queries, and lighting read identical Frame N poses and Frame N sky/IBL state. Render the first-person viewmodel directly inside Pass 4/5 Thin Prepass and Pass 8 Forward Shading (`[0.95, 1.0]` near reverse-Z sub-range, 1-bit `is_viewmodel` packed into `R8_UINT`, excluded from TLAS) so weapons receive full ReSTIR DI, reflections, and GI in an **11-pass** Render Graph.
8. **Zero Runtime PSO Stutter & Wave-Portable Subgroups:** Precompile all fixed bindless compute, mesh, and graphics pipelines (including registered `RenderEffect` pipelines) once at startup backed by `VkPipelineCache`. Write all subgroup compute shaders wave-size-agnostic (`#[spirv(subgroup_size)]`, 128-bit `SubgroupMask`) with an opt-in Wave32 pipeline hint (`subgroupSizeControl`, `requiredSubgroupSize = 32` when supported by the adapter).

---

## 2. Baseline Codebase Readiness Audit

Before Phase 0 expansion, the workspace (`ash 0.38`, `ash-window 0.13`, `gpu-allocator 0.28`, `spirv-builder 0.11.0`, `spirv-std 0.11.0`, `glam 0.34.1`, `rapier3d 0.36.1`, `winit 0.30.13`, Rust edition `2024`) has these completed foundations and remaining structural gaps:

### Ready Foundations
- **Private Vulkan 1.3 Baseline (`crates/engine/src/gpu/`):** `ash 0.38` + `ash-window 0.13` + `gpu-allocator 0.28` device/swapchain/dynamic-rendering/`synchronization2` renderer, `spirv-unknown-vulkan1.3` shader builds, `EngineError::Gpu` typed startup errors, declarative `RenderEffect` (`EffectShaderSpec`, zero `ash`/`unsafe` in `crates/fx_fresnel_rim`), private `UiRenderTarget::composite_rgba8`, and headless Vulkan 1.3 SPIR-V pipeline validation tests.
- **Simulation/Render Isolation (`crates/engine/src/runner.rs`, `snapshot.rs`):** `Renderer` is private and reads immutable double-buffered `RenderSnapshot<G>`. `RenderSchedule::Parallel` overlaps `Sim N+1` with `Render N`.
- **Shared ABI (`crates/shader_abi/src/lib.rs`):** `#![no_std]` crate with scalar-backed `#[repr(C)]` types (`Vec4f`, `Mat4f`) and optional `bytemuck` derives shared by `engine` and `core_shader`.
- **Contiguous Transform Storage (`crates/engine/src/physics.rs`):** `SoaStorage` maintains dense `positions`, `rotations`, and `bounds` arrays.
- **Decoupled UI (`crates/engine/src/ui.rs`):** `UiOverlay::render` runs after the 3D scene pass via `UiRenderTarget::composite_rgba8`.

### Structural Gaps Addressed by Phases 0 to 3
1. **Immediate-Mode Box Snapshot vs. Retained GPU Scene (`world.rs`, `gpu/scene_pipeline.rs`, `runner.rs`):** `collect_render_meshes` performs a hardcoded 200 m CPU octree query and clones box primitives every frame. Must transition to a retained GPU scene fed by a frame-index-fenced append-only monotonic delta queue (`delta.frame_index <= snapshot.frame_index`, accumulating mutations across `Sim` steps so `Parallel -> Sequential` overwrites and pre-`resumed` `GameApp::init` spawns never drop updates and `Sim N+1` never leaks into `Render N`) plus per-frame dynamic state (previous-frame transforms for motion vectors and `history_reset` invalidation flags).
2. **Missing Asset Registry & Upload Queue (`runner.rs`, `world.rs`):** `SimRunner::new` runs `GameApp::init` before `VulkanDevice` exists. Requires a device-independent opaque handle registry bridged to an async GPU upload queue.
3. **Strict Vulkan 1.4 Minimum Upgrade, Frame-Sync Migration & Render Graph (`gpu/device.rs`, `gpu/renderer.rs`, `gpu/scene_pipeline.rs`):** The completed baseline currently checks `API_VERSION_1_3` with `FRAMES_IN_FLIGHT = 2` and binary `vk::Fence`. Phase 0 upgrades startup to enforce **strict Vulkan 1.4 minimum** (`api_version >= 1.4.0`, rejecting any sub-1.4 adapter), compiles shaders to `spirv-unknown-vulkan1.4` (`multimodule(true)`), enables Vulkan 1.4 core features (via `PhysicalDeviceVulkan12Features` / `PhysicalDeviceVulkan13Features` for `runtime_descriptor_array`, `descriptor_binding_partially_bound`, `descriptor_binding_sampled_image_update_after_bind`, `descriptor_binding_variable_descriptor_count`, `shader_sampled_image_array_non_uniform_indexing`, `buffer_device_address`, `draw_indirect_count`, `subgroup_size_control`, `shader_storage_image_extended_formats`, `depth_clamp`, plus `ash 0.38` promoted KHR/EXT loaders for `push_descriptor`, `maintenance5`, `maintenance6`, `dynamic_rendering_local_read`, optional `host_image_copy`) and required extensions (`VK_KHR_ray_query`, `VK_KHR_acceleration_structure`, `VK_KHR_deferred_host_operations`, `VK_EXT_memory_budget`, optional `VK_EXT_mesh_shader`), migrates `VulkanRenderer` to triple-buffered dynamic pools + timeline-semaphore submit-serial reclamation (`completed_timeline_value >= freed_at_submit_serial`), adds optional DMA transfer/async compute queues, `VkPipelineCache`, and the internal Render Graph barrier tracker.
4. **Missing Rendering Domain Primitives (`physics.rs`, `snapshot.rs`, `save.rs`):** Needs mesh/material/skeleton handles, dynamic lights, highlight categories, cutout flags, dual `CameraMode` (First-Person + viewmodel vs. Isometric + cutout target), `VisualStyle`, `UpscalerBackend`, and asset-ID + animation-state `ActorCheckpoint` schemas.
5. **GPU-Driven Multi-Stage `RenderEffect` Evolution (`effect.rs`, `crates/fx_fresnel_rim`):** `RenderEffect` is already declarative (`EffectShaderSpec`), currently wired to CPU box vertex ranges. Phase 3 evolves `EffectShaderSpec` to typed Pass-8, Pass-10, and Pass-11 GPU-driven hook stages backed by `crates/shader_abi`, per-`EffectId` indirect/mesh queues emitted by Pass 4/5, and GPU instance `EffectAttachment` parameters.
6. **Full-World `Octree::rebuild` (`physics.rs`):** Rebuilds the entire octree on every spawn, despawn, or `SetPosition`. Must separate versioned pre-baked static sector colliders from Rapier's incremental dynamic broadphase.
7. **Large-World Precision (`physics.rs`, `snapshot.rs`, `world.rs`, `save.rs`):** `rapier3d` and `glam::Vec3` use `f32`. Seamless large worlds require canonical `DVec3` across `ActorSpawnDesc`, `PhysicsCommand`, `WorldReadView`, `SpatialQuery`, `Aabb3`, `RaycastHit`, `CameraSnapshot`, `GizmoBox`, and `ActorCheckpoint`, with active-origin rebasing for `rapier3d` and camera-relative `f32` rebasing in Pass 1.

---

## 3. Target Workspace Structure

```text
dragon-engine/
├── crates/
│   ├── shader_abi/                  # Shared CPU/GPU scalar-backed #[repr(C)] ABI structs and constants (Vec4f, Mat4f, [f32; 4]; zero glam dependency)
│   ├── asset_pipeline/              # Offline pure-Rust CLI: glTF 2.0 -> meshoptimizer-rs meshlets/LODs/HLODs, block_compression BC7/BC5 KTX2 (4x4 mip-tail padded, rayon + hash cache), fdsm MSDF fonts, versioned Rapier collider chunks, sector manifests
│   ├── engine/
│   │   ├── build.rs                 # Multi-module rust-gpu SPIR-V build (spirv-unknown-vulkan1.4, multimodule(true)) into OUT_DIR
│   │   ├── shaders/core_shader/     # 100% rust-gpu compute, mesh, ray-query, graphics, TAAU+RCAS, and MSDF shaders
│   │   └── src/
│   │       ├── gpu/                 # Private Vulkan 1.4 (ash + ash-window + gpu-allocator): device/budget probe, scaling presets, Render Graph barrier tracker, pushDescriptor Set 1, VkPipelineCache, triple-buffered pools, timeline-serial reclamation, static geometry arena, exact-budget AS arena, staging/readback rings, optional hostImageCopy, transfer/compute queues, timestamp profiler
│   │       ├── assets/              # Opaque handle registry, private OS I/O thread pool, mmap store, ref-counted shared table, per-subsystem residency policies, O(1) UPDATE_AFTER_BIND bindless mip streamer
│   │       ├── render/              # 11-pass Render Graph modules (Passes 1 through 11, including UpscalerBackend, target-res highlights, and declarative RenderEffect hook dispatch)
│   │       └── ...                  # Preserved core: clock, effect, executor, input, modes, physics, renderer, runner, save, snapshot, sync, ui, world
│   ├── fx_fresnel_rim/              # Reference declarative rust-gpu RenderEffect crate (zero ash, zero unsafe)
│   └── game/                        # Dual-camera, dual-style, seamless-streaming sample RPG
└── docs/
    ├── engine-architecture.md       # Runtime invariants
    ├── graphics-architecture.md     # Graphics and streaming Master Blueprint
    └── graphics-implementation-plan.md
```

---

## 4. Phased Implementation Plan

### Phase 0: Vulkan 1.4 Capabilities, `rust-gpu` Multi-Module Spike, Render Graph Barriers, Timeline Sync, GPU Arenas & `shader_abi`
- **Spec Sections:** 1, 7, 8, 18.
- **Dependencies:** None (builds directly on the completed `ash` + `gpu-allocator` baseline).
- **Deliverables:**
  1. **Pure `rust-gpu 0.11.0` Vulkan 1.4 Multi-Module Capability Spike (`crates/engine/build.rs`, `crates/engine/shaders/core_shader`):**
     - Migrate `build.rs` to `spirv-unknown-vulkan1.4` with `SpirvBuilder::multimodule(true)` so each entry point emits an isolated `.spv` file and optional `TaskEXT`/`MeshEXT` modules (`OpCapability MeshShadingEXT`) never pollute non-mesh fallback modules (fall back to a sibling `core_shader_mesh` crate if `spirv-builder 0.11.0` leaks module-wide capabilities).
     - Verify pure `rust-gpu` compilation and direct `vkCreateShaderModule` / pipeline creation on real Vulkan 1.4 RT hardware with `VK_LAYER_KHRONOS_validation` for: wave-size-agnostic subgroups (`#[spirv(subgroup_size)]`, 128-bit `SubgroupMask`) + opt-in Wave32 `subgroupSizeControl` hint, unsized `spirv_std::RuntimeArray<Image!(...)>` `NonUniform` bindless sampling (`UPDATE_AFTER_BIND` + `VARIABLE_DESCRIPTOR_COUNT`), Set-1 `pushDescriptor` storage buffers + 128 B push constants (`#[spirv(push_constant)]`), `RayQuery` (including non-opaque candidate-intersection alpha-test traversal), `#[spirv(early_fragment_tests)]`, `#[spirv(invariant)]`, `shaderStorageImageExtendedFormats`, and isolated `TaskEXT`/`MeshEXT` mesh shaders.
  2. **Vulkan 1.4 Device Enablement, `VK_EXT_memory_budget` Clamping, Format Probing & Dynamic Presets (`crates/engine/src/gpu/device.rs`, `config.rs`):**
     - Enforce **strict Vulkan 1.4 minimum** (`api_version >= vk::make_api_version(0, 1, 4, 0)`, rejecting any adapter below Vulkan 1.4 with `EngineError::Gpu(GpuError::UnsupportedAdapter)`) in both `new_windowed()` and single mandatory-RT `new_headless()` (real Vulkan 1.4 RT hardware always in the loop). Enable Vulkan 1.4 core features via `PhysicalDeviceVulkan12Features` / `PhysicalDeviceVulkan13Features` (`dynamic_rendering`, `synchronization2`, `timeline_semaphore`, `buffer_device_address` for host-side AS builds, `draw_indirect_count`, `subgroup_size_control`, `runtime_descriptor_array`, `descriptor_binding_partially_bound`, `descriptor_binding_sampled_image_update_after_bind`, `descriptor_binding_variable_descriptor_count`, `shader_sampled_image_array_non_uniform_indexing`, `shader_storage_image_extended_formats`, `depth_clamp`) and `ash 0.38` promoted KHR/EXT loaders (`ash::khr::push_descriptor`, `ash::khr::maintenance5`, `ash::khr::maintenance6`, `ash::khr::dynamic_rendering_local_read`, optional `ash::ext::host_image_copy`), plus required extensions (`VK_KHR_ray_query`, `VK_KHR_acceleration_structure`, `VK_KHR_deferred_host_operations`, `VK_EXT_memory_budget`), optional `VK_EXT_mesh_shader`, and optional dedicated DMA transfer + async compute queue families.
     - Probe `vkGetPhysicalDeviceFormatProperties` at startup for `VK_FORMAT_R16G16_SNORM` (`COLOR_ATTACHMENT_BIT`, falling back to `R16G16_SFLOAT`) and `VK_FORMAT_B10G11R11_UFLOAT_PACK32` (`STORAGE_IMAGE_BIT`, falling back to `R32_UINT` bit-packed or `R16G16B16A16_SFLOAT`). Return typed `EngineError::Gpu(GpuError::UnsupportedAdapter)` when Vulkan 1.4 or required ray-query/descriptor-indexing features are missing.
     - Auto-select runtime preset (`Radeon780M` at 30 FPS floor, `Radeon890M` at 60 FPS, `DiscreteDesktop` at 60+ FPS native with `0.75x` ReSTIR/GI and `0.5x` Reflections defaults + `1.0x` Quality toggle, `HighEnd4K` at 90+ FPS) clamping `STATIC_GEOMETRY_ARENA_MB` and `TEXTURE_STREAM_BUDGET_MB` dynamically via `VK_EXT_memory_budget` (`<= 75%` of available `DEVICE_LOCAL` heap budget); expose `VisualStyle` (`Realistic` | `Comic`) and `UpscalerBackend` (`TaauRcas`).
  3. **Frame-Sync Migration, Render Graph Barrier Tracker, `VkPipelineCache`, Memory Pools, Static Geometry Arena & Exact-Budget `BlasPool` (`crates/engine/src/gpu/`):**
     - Migrate `VulkanRenderer` (`gpu/renderer.rs`, `gpu/scene_pipeline.rs`) from `FRAMES_IN_FLIGHT = 2` + binary `vk::Fence` to triple-buffered dynamic pools + timeline-semaphore submit serials, driving deferred arena/BLAS/texture reclamation strictly from `completed_timeline_value >= freed_at_submit_serial`.
     - Implement lightweight internal Render Graph state tracker emitting minimal `vkCmdPipelineBarrier2` layout/access transitions per pass, binding per-pass transient resources via Set-1 `pushDescriptor`, and supporting transient scratch memory aliasing via `gpu-allocator`.
     - Implement on-disk `VkPipelineCache` load/save, triple-buffered dynamic pools, TLSF/free-list sub-allocated static geometry arena (retaining flat BLAS triangle-index slices + vertex tangent frames alongside meshlet buffers), exact-budget `BlasPool` sub-allocated inside a fixed `VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR` arena + scratch buffer (`vkGetAccelerationStructureBuildSizesKHR` + `VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR`), fixed Pass 1 upload staging ring (`STAGING_RING_MB`, DMA transfer queue capable + optional `hostImageCopy` fast path for small mip tails / UI RGBA8), and triple-buffered host-mapped mipmap-feedback readback ring.
  4. **Shared ABI Expansion (`crates/shader_abi/src/lib.rs`, Scalar-Backed `#[repr(C)]` Only) & Phase 1 Dependency Smoke Check:**
     - Keep `crates/shader_abi` free of `glam` dependencies using scalar-backed `#[repr(C)]` types (`Vec4f`, `Mat4f`, `[f32; 4]`). Define `FrameConstants` (including `viewmodel_view_proj`, `inv_viewmodel_proj`, `prev_viewmodel_view_proj`), `PushConstants` (`<= 128 B`, counts/pool discriminators/element offsets only), `GpuLight` (`32 B`), `ClusterHeader` (`8 B`), `MeshletDesc`, `GpuVertex` (position, normal, tangent, uv), `GpuSubmeshDesc` (mapping `(instance_custom_index, geometry_index)` to static/dynamic storage-buffer pool discriminator + vertex/index element offsets and `MaterialId` for Pass 7 ray-hit shading), `GpuInstance` (including `EffectAttachment` parameters and submesh range), `GpuMaterial`, `HighlightCategory`, and compressed bone transform structs.
     - Verify pure-Rust transitive dependency trees and exact APIs for Phase 1 `crates/asset_pipeline` crates (`meshoptimizer-rs`, `block_compression` `bc15`/`bc7` `compress_rgba8`, `fdsm`, `fdsm-ttf-parser`, `ktx2`, `gltf`) before starting Phase 1.
- **Verification Gate (Real Vulkan 1.4 RT Hardware with `VK_LAYER_KHRONOS_validation`):**
  - `cargo clippy --workspace --all-targets -- -D warnings`.
  - `cargo test --workspace`: `shader_abi` byte-size/alignment tests, preset/`VK_EXT_memory_budget`/arena/pool/timeline-serial/barrier-tracker unit tests, and headless Vulkan 1.4 validation of adapter probing, `RuntimeArray` `UPDATE_AFTER_BIND` + Set-1 `pushDescriptor`, push constants, wave-portable subgroups + Wave32 `subgroupSizeControl`, `#[spirv(early_fragment_tests)]`, `#[spirv(invariant)]`, `VK_KHR_ray_query`, isolated `VK_EXT_mesh_shader` `.spv` modules, optional `hostImageCopy`, and `shaderStorageImageExtendedFormats` compute storage writes.

### Phase 1: Offline Asset Pipeline (`crates/asset_pipeline`) & Async Streaming Runtime
- **Spec Sections:** 10, 11, 15, 18.
- **Dependencies:** Phase 0 (`shader_abi`, GPU arenas, staging/readback rings, `UPDATE_AFTER_BIND` descriptor set).
- **Deliverables:**
  1. **Offline Asset Compiler (`crates/asset_pipeline`, 100% Pure Rust, Zero C/C++ Build Dependencies):**
     - Ingest glTF 2.0 via pure-Rust `gltf` (meshes + MikkTSpace tangents, PBR materials, alpha-test flags, skeletons, animations, scene transforms).
     - Generate 64-vertex / 124-triangle meshlets with bounding spheres and backface cones, 4 to 6 automatic LOD levels (`build_lod_chain` and skinned `simplify_character`), and merged per-sector HLOD proxies via pure-Rust `meshoptimizer-rs` (v0.2+, zero C++ toolchain).
     - Compress albedo/ORM textures to BC7 and normal maps to BC5 via pure-Rust CPU `block_compression` (`default-features = false, features = ["bc15", "bc7"]`), padding `2x2` and `1x1` mip tails to `4x4` blocks as required by `compress_rgba8`, parallelized via `rayon` with content-hash caching, and packed into KTX2 containers via a pure-Rust KTX2 writer (`ktx2` for reading/validation).
     - Generate MSDF font atlases and glyph metric tables via pure-Rust `fdsm` + `fdsm-ttf-parser` (zero C++ `msdfgen`).
     - Pre-bake versioned serialized `rapier3d` static collider chunks (`rapier_schema_version` + raw triangle-soup/convex fallback) separating Tier-0 ground/walkmesh colliders from detailed prop colliders.
     - Emit sector containers and a reference-counted shared asset manifest.
  2. **Runtime Memory-Mapped Store & Private OS I/O Pool (`crates/engine/src/assets/mmap_store.rs`, `io_pool.rs`):**
     - Isolate `#[allow(unsafe_code)]` for read-only `mmap` inside `assets/mmap_store.rs`.
     - Run all `mmap` page reads, KTX2 mip reads, and collider deserialization on a private OS I/O thread pool (zero I/O in `TaskSlab` or render thread) with instant stale-request cancellation and lock-free handoff compatible with `RenderSchedule::Parallel`.
  3. **Device-Independent Handle Registry & Independent Residency Coordinators (`crates/engine/src/assets/`):**
     - Expose opaque handles (`MeshHandle`, `MaterialHandle`, `TextureHandle`, `SkeletonHandle`, `FontHandle`).
     - Implement independent residency policies for Physics (`PHYSICS_ACTIVE_RADIUS_M`), Geometry/LOD (`CAMERA_FAR_M`), Actors (`SIM_ACTIVE_RADIUS_M`), plus CPU-side residency state machines and unit tests for Ray-Tracing BLAS (`TLAS_ACTIVE_RADIUS_M + BLAS_PREFETCH_MARGIN_M`, wired to Pass 3 in Phase 4) and `BindlessTextureTable` mipmap feedback (`VK_EXT_memory_budget`-clamped `TEXTURE_STREAM_BUDGET_MB`, 16,384 to 65,536 `UPDATE_AFTER_BIND` slots, generational slots, O(1) `vkUpdateDescriptorSets` per promoted/demoted `VkImage`, atomic `min_lod` clamp, timeline-serial deferred destroy, and host-mapped feedback drain wired to Pass 8 in Phase 3).
     - Implement player-gravity-only fallback when Tier-0 ground collider is not yet resident.
- **Verification Gate:**
  - Offline CLI compiles an animated multi-material glTF test scene and font into sector blobs (all meshlets `<= 64` verts and `<= 124` tris, 4+ LODs + HLOD, BC7/BC5 KTX2 including `2x2`/`1x1` mip tails, MSDF atlas, versioned Rapier chunks).
  - Async runtime streaming stress test confirming `0.00 ms` main-thread blocking disk I/O, clean ref-counted eviction under `TEXTURE_STREAM_BUDGET_MB`, and zero stale-request leaks.

### Phase 2: Retained Scene Boundary, Physics Refactor, Reverse-Z, Two-Phase Hi-Z, Thin Prepass (World + Viewmodel) & Dual-Mode Far Sun Cascade (Passes 1, 4, 5)
- **Spec Sections:** 3, 4, 5, 6, 10, 16.
- **Dependencies:** Phase 0, Phase 1.
- **Deliverables:**
  1. **Retained Scene & `RenderSnapshot` Evolution (`snapshot.rs`, `world.rs`, `save.rs`, `runner.rs`, `gpu/renderer.rs`):**
     - Replace per-frame 200 m CPU box collection (`RenderSnapshot::meshes` migration across `crates/engine` tests and `crates/game`) with retained GPU scene handles fed by an append-only monotonic delta queue fenced strictly by `delta.frame_index <= snapshot.frame_index` (accumulating mutations across `Sim` steps so `Parallel -> Sequential` overwrites and pre-`resumed` `GameApp::init` spawns never drop updates, while preventing concurrent `Sim N+1` mutations from bleeding into `Render N`) plus per-frame dynamic state (camera mode, `history_reset` invalidation flag, previous-frame transforms for motion vectors, dynamic lights, skeleton poses, highlight actor list + categories, viewmodel state, `EffectAttachment` parameters, `VisualStyle`, `GizmoBox`). Drain fenced deltas and latch `history_reset` inside `VulkanRenderer::draw` before `acquire_next_image` early-returns so `OUT_OF_DATE`/`SUBOPTIMAL` frames never drop GPU scene state.
     - Implement sector actor hibernation/restoration via `ActorCheckpoint` (including skeletal animation state) into an in-memory Sector Delta Store.
  2. **Physics Refactor & Large-World `DVec3` Rebasing (`physics.rs`, `world.rs`, `snapshot.rs`, `save.rs`):**
     - Remove full-world `Octree::rebuild` on dynamic moves; back `SpatialQuery::query_octree` with Rapier's incremental broadphase/query pipeline and splice versioned pre-baked static sector colliders cleanly.
     - Migrate canonical CPU world coordinates across `ActorSpawnDesc`, `PhysicsCommand`, `WorldReadView`, `SpatialQuery`, `Aabb3`, `RaycastHit`, `CameraSnapshot`, `GizmoBox`, and `ActorCheckpoint` to `DVec3` with active physics-origin rebasing so `f32` `rapier3d` remains precise across large seamless worlds.
  3. **Pass 1 Camera-Relative Reverse-Z & Dual Camera:**
     - Reverse-Z `VK_FORMAT_D32_SFLOAT` (`1.0` near, `0.0` far, `GREATER_OR_EQUAL`, clear `0.0`, negative Vulkan viewport height), camera-relative `f32` rebasing, First-Person (`0.05 m` to `2,000 m`, plus dedicated near reverse-Z viewmodel projection `[0.95, 1.0]`) and Isometric (`10 m` to `80 m`) modes, Halton jitter toggle (on for Realistic temporal upscaling, zero for Comic).
  4. **Passes 4 and 5 (Two-Phase Hi-Z Culling, Unified Thin Prepass & Dual-Mode Far Sun Cascade):**
     - Phase 1 compute meshlet culler (frustum, backface cone, screen-error LOD, prev-frame Hi-Z skipped on `history_reset`, wave-portable subgroup compaction with opt-in Wave32 hint, solid vs. alpha-test vs. viewmodel vs. per-`EffectId` Pass-8 overlay split, isometric camera-to-player capsule/cone wall cutout routing with hysteresis).
     - Thin prepass (binding the global bindless texture set for alpha-test discard and BC5 normal maps, marking clip-space `Position` `#[spirv(invariant)]`) writing internal-resolution reverse-Z `D32_SFLOAT`, BC5 normal-mapped octahedral world normals (`R16G16_SNORM` or `R16G16_SFLOAT` fallback), packed roughness + viewmodel flag (`R8_UINT`: 7-bit roughness + 1-bit `is_viewmodel`), and rigid-body + viewmodel motion vectors (`R16G16_SFLOAT`; extended to skinned limb motion vectors in Phase 4) totaling `9 B/sample` across 3 color attachments (zero `R16_UINT` Object ID overhead in Pass 4) via isolated `VK_EXT_mesh_shader` `.spv` modules or `vkCmdDrawIndexedIndirectCount` fallback. Render first-person viewmodel hands/weapons in the `[0.95, 1.0]` depth sub-range with `is_viewmodel = 1` and viewmodel motion vectors (`prev_viewmodel_view_proj`).
     - Sun-frustum cull and `2048x2048` directional Dual-Mode Far Sun Shadow Cascade raster pass (`depthClamp` enabled): tight active orthographic frustum in Isometric mode (for Pass 10 cutout walls) vs. far-vista coverage beyond `TLAS_ACTIVE_RADIUS_M` in First-Person mode.
     - Current-frame Hi-Z `min`-depth pyramid build (`R32_SFLOAT`), Phase 2 disoccluded meshlet pass, and merging Phase 1 + Phase 2 visible meshlets (+ viewmodel) into Pass-8 replay indirect buffers.
- **Verification Gate:**
  - 10,000-meshlet behind-wall occlusion test (>95% hidden culled, zero popping on rapid camera rotation or `history_reset`).
  - Simultaneous `0.05 m` and `2,000 m` reverse-Z depth precision test, first-person flush-wall viewmodel anti-clip check, large-world `DVec3` origin-rebase test, and `RenderSchedule::Parallel` frame-fenced delta coherence test.

### Phase 3: Compute 3D Light Clustering, Dual-Style Opaque Forward Shading & Declarative GPU-Driven `RenderEffect` Hooks (Passes 6, 8)
- **Spec Sections:** 2, 3, 7, 8, 11, 16.
- **Dependencies:** Phase 2.
- **Deliverables:**
  1. **Pass 6 (Active-Cluster Depth Marking & Compute 3D Cluster Light Assignment):**
     - Exponential 3D cluster grid (`32` to `64` per axis, recomputed on resize/FOV/camera-mode change).
     - Lightweight depth-marking compute dispatch reading Pass 5 reverse-Z depth and `is_viewmodel` (unprojecting viewmodel pixels via `inv_viewmodel_proj`) to flag occupied screen clusters before light assignment.
     - Wave-size-agnostic subgroup ballot/shuffle/barrier light assignment (opt-in Wave32 hint when supported) for up to `4,096` dynamic `GpuLight` entries (`32 B` each) strictly over active occupied clusters.
  2. **Pass 8 (Opaque Clustered Forward Shading for World + Viewmodel & Declarative Pluggable `RenderEffect` Hooks):**
     - Zero-overdraw `Depth == Equal` pass (`depth_write_enable = false`, `#[spirv(early_fragment_tests)]`, `#[spirv(invariant)]`) replaying the merged Pass 4+5 visible meshlet lists (including first-person viewmodel) and writing `VK_FORMAT_B10G11R11_UFLOAT_PACK32` linear HDR color, sampling `RuntimeArray<Image!(...)>` bindless PBR textures and writing mipmap feedback requests.
     - Dual runtime-selectable shader entry points: **Realistic PBR** (multiplying Pass 7 demodulated ReSTIR DI diffuse/specular irradiance, glossy reflections, and Radiance Cascades GI by material albedo/F0/BRDF, with temporary clustered direct fallback before Phase 5 lands, plus Frame N Dynamic Time-of-Day Skybox IBL and built-in material Fresnel rim) and **Flat Comic** (quantized cel bands, flat ambient, disabled SH/specular reflections, optional Fresnel rim).
     - Extend declarative `EffectShaderSpec` (`crates/engine/src/effect.rs`) with typed GPU-driven hook stages (Pass 8 opaque material/overlay hooks reading per-`EffectId` visible meshlet queues + GPU instance `EffectAttachment` parameters via engine-provided mesh/vertex front-ends, Pass 10 transparent hooks, Pass 11 screen-space post hooks) while keeping `crates/fx_fresnel_rim` 100% free of `ash` and `unsafe`.
     - Migrate `crates/fx_fresnel_rim` to the GPU-driven Pass-8 `EffectShaderSpec` contract as the reference custom shader plugin crate.
- **Verification Gate:**
  - 4,096-light stress benchmark: Pass 6 within `0.10 to 0.40 ms` design reference on target desktop GPU, zero visible cluster boundary seams.
  - Runtime Realistic/Comic switch and declarative `crates/fx_fresnel_rim` custom `rust-gpu` hook verified on identical glTF assets with `VK_LAYER_KHRONOS_validation`.

### Phase 4: Skeletal Animation Evaluation, Compute Meshlet Skinning, Dynamic TOD Sky/IBL Compute & Unified Hardware BVH (Passes 2, 3)
- **Spec Sections:** 9, 12, 13, 14.
- **Dependencies:** Phase 2.
- **Deliverables:**
  1. **Skeletal Animation Evaluation (`Sim` / `WorkerPool`):**
     - Sample glTF animation clips, blend poses, and compute hierarchical bone palettes into `RenderSnapshot` dynamic deltas; serialize clip ID, normalized playback time, and blend weights in `ActorCheckpoint`.
  2. **Pass 2 (Compute Meshlet Skinning + Skinned Motion Vectors + Dynamic TOD Sky LUT / Amortized IBL Compute):**
     - Upload compressed bone matrices to 12 MB triple-buffered storage pool.
     - Wave-size-agnostic subgroup compute skinning (opt-in Wave32 hint when supported) writing Frame N deformed positions, normals, and tangents into the skinned vertex pool (`SHADER_DEVICE_ADDRESS | ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY`) while retaining previous-frame skinned positions (`36 to 72 MB` total) so Pass 4 emits accurate per-vertex motion vectors for animated limbs.
     - Dynamic Time-of-Day physical sky atmosphere LUT (`4 MB`) update + amortized compute re-bake of the 16 MB Skybox IBL cubemap (prefiltered specular mips + SH diffuse irradiance) scheduled in Pass 2 ahead of Pass 7 (overlappable on async compute with Pass 4/5, guaranteed initialized on Frame 0; skipped in Comic mode) so Passes 7, 8, 9, and 10 read identical Frame N sky/IBL state.
  3. **Pass 3 (In-Place Dynamic BLAS Refit, Async Static BLAS Compaction & Active-Radius TLAS):**
     - `synchronization2` memory barrier from Pass 2 skinned vertex writes to acceleration structure build reads.
     - Execute true hardware in-place refit (`ALLOW_UPDATE | PREFER_FAST_TRACE`, `MODE_UPDATE`) on active dynamic character BLASes from Frame N skinned vertices (up to `MAX_DYNAMIC_BLAS_COUNT`).
     - Execute up to `STATIC_BLAS_BUILDS_PER_FRAME` queued static BLAS builds and drive the multi-frame async compaction state machine (`vkCmdWriteAccelerationStructuresPropertiesKHR` -> timeline retire -> `vkCmdCopyAccelerationStructureKHR COMPACT` -> atomic TLAS swap -> timeline-serial arena free) inside the exact `BLAS_POOL_BUDGET_MB` arena.
     - Rebuild single shared camera-relative TLAS for `RayReady` objects inside `TLAS_ACTIVE_RADIUS_M` (excluding the first-person viewmodel to prevent self-shadowing, and retaining isometric cutout walls as opaque shadow casters), setting 24-bit `instanceCustomIndex` to each object's `GpuSubmeshDesc` range and setting `VkAccelerationStructureInstanceKHR::flags` (`FORCE_NO_OPAQUE` for near-range alpha-tested foliage inside `ALPHA_TEST_RT_RANGE_M`, `FORCE_OPAQUE` beyond `ALPHA_TEST_RT_RANGE_M`).
- **Verification Gate:**
  - Animated character ray-hit validation confirming `0`-frame lag between rasterized limbs and BVH hits (zero bind-pose spawn flash) and clean limb motion vectors in Pass 4.
  - Frame-0 and time-of-day jump validation confirming Pass 7/8/9/10 read identical Frame N Skybox IBL.
  - 100-character timing benchmark on target desktop GPU verifying Pass 2 skinning, in-place dynamic BLAS refit (`0.15 to 0.35 ms` reference), async static BLAS compaction, and TLAS rebuild.

### Phase 5: Unified Hardware Ray Queries (Demodulated ReSTIR DI, Reflections, Radiance Cascades GI) & Dual-Mode Denoiser (Pass 7)
- **Spec Sections:** 4, 12, 17.
- **Dependencies:** Phase 3, Phase 4.
- **Deliverables:**
  1. **Demodulated ReSTIR DI + Dual-Mode Far Sun Cascade Lookup:**
     - Shared `reconstruct_world_pos(uv, depth, is_viewmodel)` helper (using `inv_viewmodel_proj` for first-person weapons so hands and metallic weapons receive full shadows, reflections, and GI without self-shadowing).
     - Automatic cluster-light + sun stochastic reservoir resampling (`TERMINATE_ON_FIRST_HIT` rays against shared TLAS; non-opaque candidate-intersection alpha-test inside `ALPHA_TEST_RT_RANGE_M`, opaque proxy via `FORCE_OPAQUE` beyond), blending seamlessly into Pass 4's `2048x2048` directional Far Sun Cascade beyond `TLAS_ACTIVE_RADIUS_M`, and writing **Denoised Direct Diffuse Irradiance** (`B10G11R11_UFLOAT_PACK32`) and **Denoised Direct Specular Irradiance** (`B10G11R11_UFLOAT_PACK32`) for Pass 8 albedo/BRDF modulation.
  2. **Hit-Shaded Ray-Queried Glossy Reflections:**
     - 1 stochastic GGX ray per pixel where unpacked roughness `< REFLECTION_ROUGHNESS_CUTOFF` (`0.5x` res default on Radeon 780M, Radeon 890M, and `DiscreteDesktop`; `1.0x` Quality toggle and `HighEnd4K` default; disabled in Comic). On hit, resolve `(instance_custom_index, geometry_index)` through `GpuSubmeshDesc` (bound storage buffers + static/dynamic pool discriminator + element offsets) -> flat BLAS vertex/skinned slice + index slice + `MaterialId` -> bindless albedo/ORM textures and evaluate hit-point lighting (with Frame N Skybox IBL on sky misses). Zero SSAO/GTAO passes.
  3. **Hardware Ray-Queried Radiance Cascades:**
     - Screen-space depth-anchored Cascade 0 (`1.5 m` to `2.0 m` short rays) transitioning to coarse 3D camera-centered clipmap higher cascades against the shared TLAS (zero 3D SDFs; `0.5x` on iGPUs, `0.75x` default on `DiscreteDesktop`, `1.0x` Quality/`HighEnd4K`), using `(instance_custom_index, geometry_index)` hit-point lookup + previous-frame merged radiance cache for leak-free multi-bounce GI.
  4. **Dual-Mode Denoiser:**
     - Temporal-spatial A-Trous filter respecting `is_viewmodel` boundary edge-stopping (`weight = 0` across seam) using Pass 4 motion vectors and `history_reset` in Realistic mode (`~40 to 48 MB` history + ReSTIR reservoirs at 1080p); spatial-only normal/depth-guided filter in Comic mode (`0 MB` history).
- **Verification Gate:**
  - Multi-colored overlapping torch/spell shadow test confirming zero cross-light shadow bleed (Demodulated ReSTIR DI).
  - Thin-wall cave/interior + first-person metallic weapon test verifying zero GI light leakage, true corner occlusion without SSAO, and accurate RT shadows/reflections/GI on animated characters and weapons.
  - Offline QA profiler pass on target discrete desktop GPU (and Radeon 780M floor) confirming Pass 7 stays within design reference budgets under `VK_LAYER_KHRONOS_validation`.

### Phase 6: Sky Fill/Fog, Transparent GPU Radix Sort, Pluggable Upscaling, Fragment Tonemapping, Target-Res Highlights & MSDF Text (Passes 9, 10, 11)
- **Spec Sections:** 5, 13, 15, 16.
- **Dependencies:** Phase 3, Phase 5.
- **Deliverables:**
  1. **Pass 9 (Sky Background Fill & Opaque Height Fog):** Draw physical sky from Pass 2 sky atmosphere LUT + opaque height fog (skipping pixels where `is_viewmodel == 1`) in Realistic mode; draw stylized flat gradient sky in Comic mode where reverse-Z depth == `0.0`.
  2. **Pass 10 (Transparent Meshlet GPU Subgroup Radix Sort, Alpha Blend & Pluggable Overlay Hooks):** Camera-depth key generation, 3 to 4 dispatch back-to-front wave-size-agnostic GPU subgroup radix sort (opt-in Wave32 hint when supported, `1` to `4 MB`), single `vkCmdDrawMeshTasksIndirectCountEXT` / `vkCmdDrawIndexedIndirectCount` hardware alpha blend pass depth-tested against Pass 5 depth (including viewmodel near depth), evaluating Pass 6 clustered lights, Frame N Skybox IBL, per-fragment height fog, and Pass 4 Dual-Mode Far Sun Shadow Cascade lookup on isometric cutout walls, plus Pass-10 `RenderEffect` hooks.
  3. **Pass 11 (Pure `rust-gpu` `UpscalerBackend`, Fragment Tonemapping, Post Hooks, Target-Res Highlights, Debug Gizmos, MSDF Text & `UiOverlay`):**
     - **Unified `UpscalerBackend` (`TaauRcas`):** 100% `rust-gpu` temporal anti-aliasing upscaler with Halton jitter, `history_reset`, `is_viewmodel` boundary protection, and RCAS sharpening writing target-resolution HDR, followed by a fullscreen AgX tonemap fragment pass writing directly into the sRGB swapchain image.
     - Comic mode: zero temporal history/jitter (`0.0 MB`), spatial scaling (EASU/bilinear + RCAS when `INTERNAL_RENDER_SCALE < 1.0`, or native `1.0` scale) + custom 3D color LUT + depth/octahedral-normal geometry ink outlines (checking `is_viewmodel` to prevent false depth-edge bleed between weapon and world).
     - Registered Pass-11 `RenderEffect` post-process hooks.
     - **Post-Tonemap Target-Resolution Highlight Pass (Option 4B):** Un-jittered pass skipped completely when highlighted actors == 0 (`0.00 ms`); otherwise renders only highlighted actors into a transient aliased target-resolution `R8_UINT` mask depth-tested against 3x3 dilated upsampled scene depth to composite native 1-pixel post-AgX tactical outlines, optional occluded silhouettes behind walls, and animated loot reveal patterns.
     - Post-tonemap `GizmoBox` debug overlays, GPU MSDF text renderer (3D world signs + HUD hooks), and `UiOverlay::render` (`UiRenderTarget::composite_rgba8`, optionally accelerated via Vulkan 1.4 `hostImageCopy`).
- **Verification Gate:**
  - Isometric concave L-wall cutout depth-order, Far Sun shadow consistency, and hysteresis test (zero manual artist mesh splitting).
  - First-person flush-wall weapon anti-clip, metallic RT reflection, and temporal-ghosting check.
  - Camera-mode switch (`FirstPerson <-> Isometric`), window resize, and style switch (`Realistic <-> Comic`) verifying clean `history_reset` with zero temporal ghosting.
  - Post-AgX 1-pixel highlight sharpness, occluded silhouette, and zero-shimmer check at 720p->1080p and 1440p->4K, plus MSDF crispness check at 720p, 1080p, 4K, and close-up 3D world signs.

### Phase 7: Sample Game Integration, Vulkan GPU Timestamp Profiler & Documentation Sync
- **Spec Sections:** All (1 through 18).
- **Dependencies:** Phase 6.
- **Deliverables:**
  1. **Dual-Camera, Dual-Style, Seamless-Streaming Sample (`crates/game`):**
     - Runtime camera mode switch (First-Person with unified RT-lit viewmodel vs. Isometric with concave wall cutouts).
     - Runtime visual style switch (`Realistic` vs. `Comic`) and declarative `crates/fx_fresnel_rim` pluggable `RenderEffect` demonstration.
     - Multi-sector seamless traversal demonstrating async pop-in, animated glTF characters, overlapping colored torches/spells (ReSTIR DI), glossy floors, alpha-tested foliage, glass/water, post-AgX 1-pixel + occluded highlights, Fresnel rims, and 3D MSDF signs using pure reactive `GameApp` state.
  2. **11-Pass Vulkan GPU Timestamp Profiler (`crates/engine/src/gpu/profiler.rs`):**
     - Per-pass `VK_QUERY_TYPE_TIMESTAMP` (`vkCmdWriteTimestamp2`, scaled by `timestampPeriod`) across all 11 passes calibrated against the target discrete desktop GPU tier and Radeon 780M floor.
  3. **Documentation & Skills Sync (`docs/engine-architecture.md`, `.agents/skills/`):**
     - Align runtime diagram, Vulkan 1.4 notes, declarative `RenderEffect` hooks, and streaming subsystems with the completed 11-pass renderer.
- **Verification Gate (Real Vulkan 1.4 RT Hardware):**
  - `cargo clippy --workspace --all-targets -- -D warnings`.
  - `cargo test --workspace` with `VK_LAYER_KHRONOS_validation` + GPU-AV (zero validation errors).
  - Verified 11-pass GPU timestamp report calibrated on target discrete desktop hardware and Radeon 780M floor.

---

## 5. 5-Person Team Parallelization Map

After Phase 0 establishes Vulkan 1.4 capabilities, multi-module SPIR-V builds, Render Graph barriers, `pushDescriptor`, timeline-serial sync, GPU arenas, presets, and `shader_abi`, work runs in parallel across five ownership tracks:

| Engineer | Ownership | Phases |
| :--- | :--- | :--- |
| **Engineer 1 (GPU Core & Geometry)** | `gpu/` Render Graph Barriers, `pushDescriptor` & `VkPipelineCache`, Reverse-Z, Two-Phase Hi-Z, Meshlet Culling, Unified Thin Prepass (World + Viewmodel), Dual-Mode Far Sun Cascade, Transparent Radix Sort | 0, 2, 6 (Pass 10) |
| **Engineer 2 (Ray Tracing & GI)** | Phase 0 `RayQuery` Spike & Exact-Budget `BlasPool` Arena, In-Place Dynamic Refit, Async Compaction, Active-Radius TLAS, Demodulated ReSTIR DI, Hit-Shaded Reflections, Radiance Cascades, Dual-Mode Denoiser | 0, 4 (Pass 3), 5 (Pass 7) |
| **Engineer 3 (Shading, Lighting, Upscaling & Effects)** | Phase 0 Multi-Module `RuntimeArray`/Wave-Portable Subgroup Spike & scalar-backed `shader_abi`, Active-Cluster Depth Mark & 3D Light Clustering, PBR/Comic Forward Shaders, Declarative GPU-Driven `RenderEffect` Hooks (`crates/fx_fresnel_rim`), Dynamic TOD Sky/IBL Compute (Pass 2) & Sky Fill/Fog (Pass 9), Pure `rust-gpu` `TaauRcas`, AgX/Comic LUT, Target-Res Post-AgX Highlights, Post-Tonemap Gizmos | 0, 3 (Passes 6, 8), 4 (Pass 2 Sky/IBL), 6 (Passes 9, 11) |
| **Engineer 4 (Assets, Streaming, Skinning & Text)** | Phase 0 Arena/Feedback Ring ABI & Phase 1 Crate Smoke Check, `crates/asset_pipeline`, `mmap` I/O Pool, Independent Residency Policies, `UPDATE_AFTER_BIND` `BindlessTextureTable` Mip Streamer (`hostImageCopy` fast path), Skeletal Eval & Compute Skinning, MSDF Text | 0, 1, 4 (Pass 2 Skinning), 6 (MSDF) |
| **Engineer 5 (Runtime, Physics, Game & QA)** | Phase 0 Vulkan 1.4 Device/Budget Startup & Timeline-Serial Migration, Frame-Fenced Retained Scene / Pre-Acquire Delta Drain / `RenderSnapshot`, `physics.rs` Sector Colliders & `DVec3` Origin Rebasing, Sector Actor Hibernation, `crates/game` Showcase, 11-Pass Vulkan Timestamp Profiler & Real-HW Validation Suite | 0, 2, 7 |

---

## 6. Phase Progress Tracker

| Phase | Scope | Status |
| :--- | :--- | :--- |
| **Baseline** | Private Vulkan 1.3 (`ash 0.38` + `ash-window 0.13` + `gpu-allocator 0.28`), `spirv-unknown-vulkan1.3`, declarative `RenderEffect`, private `UiRenderTarget::composite_rgba8`, typed `EngineError` | Completed |
| **Phase 0** | Vulkan 1.4 target + `spirv-unknown-vulkan1.4` `multimodule(true)` RT/Mesh/Bindless/Wave32 spike, `VK_EXT_memory_budget` clamping, `pushDescriptor`/`hostImageCopy`, timeline-serial sync migration, Render Graph barrier tracker, `VkPipelineCache`, static geometry & exact AS arenas, staging/readback rings, `shader_abi`, Phase 1 crate smoke check | Not Started |
| **Phase 1** | `crates/asset_pipeline` (versioned colliders, `4x4`-padded BC7/BC5 KTX2, `rayon` + hash cache), `mmap` OS I/O pool, independent residency policies, O(1) `UPDATE_AFTER_BIND` bindless mip streamer | Not Started |
| **Phase 2** | Frame-fenced retained scene boundary, `physics.rs` refactor & full `DVec3` origin rebasing, Reverse-Z, Two-Phase Hi-Z, Unified 9 B/sample Thin Prepass (World + Viewmodel, `#[spirv(invariant)]`) & Dual-Mode Far Sun Cascade (Passes 1, 4, 5) | Not Started |
| **Phase 3** | Compute 3D light clustering (wave-portable + Wave32 hint), dual-style opaque forward shading (World + Viewmodel) & declarative GPU-driven `RenderEffect` hooks (`crates/fx_fresnel_rim`) (Passes 6, 8) | Not Started |
| **Phase 4** | Skeletal eval, compute meshlet skinning (+ prev-frame positions), Pass-2 Dynamic TOD Sky LUT / amortized IBL compute & Pass-3 in-place dynamic BLAS refit / async static compaction / TLAS pipeline (Passes 2, 3) | Not Started |
| **Phase 5** | Unified hardware ray queries (Demodulated ReSTIR DI diffuse/specular, hit-shaded reflections at 0.5x `DiscreteDesktop` default, Radiance Cascades GI at 0.75x `DiscreteDesktop` default, viewmodel RT shading) & dual-mode denoiser (Pass 7) | Not Started |
| **Phase 6** | Sky fill/fog, transparent radix sort (+ per-fragment lighting/fog + Dual-Mode Far Sun shadow), pure `rust-gpu` `UpscalerBackend` (`TaauRcas`), fragment tonemap, post-AgX target-res 1-px highlights & MSDF text (Passes 9, 10, 11) | Not Started |
| **Phase 7** | `crates/game` showcase, 11-pass Vulkan timestamp profiler (`vkCmdWriteTimestamp2`) & `docs/engine-architecture.md` / `.agents/skills/` sync | Not Started |
