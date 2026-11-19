# Implementation Plan: Dual-Camera 3D RPG Graphics & Streaming Architecture

This document defines the phased implementation plan for realizing [Architectural Design Specifications for a Dual-Camera 3D RPG Graphics Engine](graphics-architecture.md) inside Dragon Engine while preserving all runtime invariants in [Engine Architecture Principles](engine-architecture.md).

---

## 1. Locked Architectural Decisions

These decisions are final and must not be reopened during implementation without an explicit Blueprint review:

1. **Vulkan 1.3 Only via Private `ash 0.38` + `gpu-allocator 0.28`:** Target native **Vulkan 1.3** exclusively (Linux including Steam Deck, and Windows) through private `ash 0.38`, `ash-window 0.13`, and pure-Rust `gpu-allocator 0.28` inside `crates/engine/src/gpu/`. Native macOS, DX12, Metal, WebAssembly, and `wgpu` are out of scope. Zero raw Vulkan handles are exposed outside `crates/engine`.
2. **Strict 100% `rust-gpu` Vulkan 1.3 SPIR-V (Zero WGSL, Zero Shader Cross-Compilers):** All engine-authored and game-authored compute, mesh, ray-query, graphics, upscaling (`TaauRcas`), and declarative `RenderEffect` shaders live in `crates/engine/shaders/core_shader` and external `rust-gpu` crates (such as `crates/fx_fresnel_rim`) compiled with `spirv-builder 0.11.0` (`spirv-unknown-vulkan1.3`) into Cargo `OUT_DIR` and loaded directly via `vkCreateShaderModule`. All runtime and `crates/asset_pipeline` crates use strictly pure-Rust dependencies (with `spirv-builder`'s transitive `spirv-tools-sys` as the sole shader-build exception).
3. **Scoped `unsafe` Isolation:** Keep `unsafe_code = "deny"` in root `Cargo.toml` across all public APIs, simulation, gameplay, `shader_abi`, `asset_pipeline`, and `fx_fresnel_rim`. Isolate `#[allow(unsafe_code)]` strictly inside two private audited boundaries with explicit `// SAFETY:` comments and mandatory `VK_LAYER_KHRONOS_validation` testing:
   - `crates/engine/src/gpu/` (private Vulkan 1.3 `ash` + `gpu-allocator` backend), and
   - `crates/engine/src/assets/mmap_store.rs` (read-only immutable `mmap` slices).
4. **Zero-Loading-Screen Async Pop-In (Frame-Pacing First):** Never stall simulation or rendering on disk I/O, decompression, GPU upload, BLAS build/compaction, or physics splicing. Accept visible asynchronous pop-in over frame-rate stutter.
5. **Independent Per-Subsystem Residency Policies:** Physics, Geometry/LOD, Ray-Tracing BLAS, Actors/AI, and GPU-Feedback Mipmap Streaming each own independent spatial residency policies and scaling parameters (no coupled monolithic sector rings).
6. **Lightweight Streaming Pacing (No Runtime Millisecond Timers):** Use only a fixed-size Pass 1 staging ring (`STAGING_RING_MB`), a runtime-preset per-frame static BLAS build cap (`STATIC_BLAS_BUILDS_PER_FRAME`), and explicit pool/VRAM budgets (`STATIC_GEOMETRY_ARENA_MB`, `BLAS_POOL_BUDGET_MB`, `TEXTURE_STREAM_BUDGET_MB`). Millisecond figures are design-time references and offline QA profiler targets, never runtime timers.
7. **Zero-Lag Pass 2 / Pass 3 Ordering:** Execute **Pass 2 (Compute Meshlet Skinning)** before **Pass 3 (In-Place Dynamic BLAS Refit, Async Static BLAS Compaction, and Active-Radius TLAS Rebuild)** so rasterization and hardware ray queries consume identical Frame N character geometry, while Pass 2 retains previous-frame skinned positions for Pass 4 motion vectors.
8. **Zero Runtime PSO Stutter:** Precompile all fixed bindless compute, mesh, and graphics pipelines (including registered `RenderEffect` pipelines) once at startup backed by `VkPipelineCache`.

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
3. **Extended Vulkan 1.3 Feature Enablement & Render Graph (`gpu/device.rs`, `gpu/`):** The baseline enables core `dynamic_rendering`, `synchronization2`, and `timeline_semaphore`. Phase 0 enables `VK_KHR_ray_query`, `VK_KHR_acceleration_structure`, `VK_KHR_deferred_host_operations`, `VK_EXT_mesh_shader` (optional), `VK_EXT_descriptor_indexing` (`runtimeDescriptorArray`, `UPDATE_AFTER_BIND`, `PARTIALLY_BOUND`, `VARIABLE_DESCRIPTOR_COUNT`, non-uniform indexing), `shaderStorageImageExtendedFormats`, `depthClamp`, `drawIndirectCount`, `pushConstants`, optional DMA transfer/async compute queues, `VkPipelineCache`, and the internal Render Graph barrier tracker.
4. **Missing Rendering Domain Primitives (`physics.rs`, `snapshot.rs`, `save.rs`):** Needs mesh/material/skeleton handles, dynamic lights, highlight categories, cutout flags, dual `CameraMode` (First-Person + viewmodel vs. Isometric + cutout target), `VisualStyle`, `UpscalerBackend`, and asset-ID + animation-state `ActorCheckpoint` schemas.
5. **GPU-Driven Multi-Stage `RenderEffect` Evolution (`effect.rs`, `crates/fx_fresnel_rim`):** `RenderEffect` is already declarative (`EffectShaderSpec`), currently wired to CPU box vertex ranges. Phase 3 evolves `EffectShaderSpec` to typed Pass-8, Pass-10, and Pass-12 GPU-driven hook stages backed by `crates/shader_abi`, per-`EffectId` indirect/mesh queues emitted by Pass 4/5, and GPU instance `EffectAttachment` parameters.
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
│   │   ├── build.rs                 # Multi-entry rust-gpu SPIR-V build (spirv-unknown-vulkan1.3) into OUT_DIR
│   │   ├── shaders/core_shader/     # 100% rust-gpu compute, mesh, ray-query, graphics, TAAU+RCAS, and MSDF shaders
│   │   └── src/
│   │       ├── gpu/                 # Private Vulkan 1.3 (ash + ash-window + gpu-allocator): device probe, scaling presets, Render Graph barrier tracker, VkPipelineCache, triple-buffered pools, static geometry arena, exact-budget AS arena, staging/readback rings, transfer/compute queues, timestamp profiler
│   │       ├── assets/              # Opaque handle registry, private OS I/O thread pool, mmap store, ref-counted shared table, per-subsystem residency policies, O(1) UPDATE_AFTER_BIND bindless mip streamer
│   │       ├── render/              # 12-pass Render Graph modules (Passes 1 through 12, including UpscalerBackend and declarative RenderEffect hook dispatch)
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

### Phase 0: Vulkan 1.3 Capabilities, `rust-gpu` Capability Spike, Render Graph Barriers, GPU Arenas & `shader_abi`
- **Spec Sections:** 1, 7, 8, 18.
- **Dependencies:** None (builds directly on the completed `ash` + `gpu-allocator` baseline).
- **Deliverables:**
  1. **Pure `rust-gpu 0.11.0` Vulkan 1.3 Capability Spike (`crates/engine/shaders/core_shader`):**
     - Verify pure `rust-gpu` compilation and direct `vkCreateShaderModule` / pipeline creation on Vulkan 1.3 with `VK_LAYER_KHRONOS_validation` for: subgroups and subgroup barriers (`GL_KHR_shader_subgroup`), unsized `spirv_std::RuntimeArray<Image!(...)>` `NonUniform` bindless sampling (`UPDATE_AFTER_BIND` + `VARIABLE_DESCRIPTOR_COUNT`), 128 B push constants (`#[spirv(push_constant)]`), `RayQuery` (including non-opaque candidate-intersection alpha-test traversal), `#[spirv(early_fragment_tests)]`, `VK_KHR_buffer_device_address`, and `TaskEXT`/`MeshEXT` mesh shaders.
  2. **Extended Vulkan 1.3 Device Enablement, Format Probing & Dynamic Presets (`crates/engine/src/gpu/device.rs`, `config.rs`):**
     - Enable `VK_KHR_ray_query`, `VK_KHR_acceleration_structure`, `VK_KHR_deferred_host_operations`, `VK_KHR_buffer_device_address`, `VK_EXT_descriptor_indexing` (`runtimeDescriptorArray`, `descriptorBindingPartiallyBound`, `descriptorBindingSampledImageUpdateAfterBind`, `descriptorBindingVariableDescriptorCount`, `shaderSampledImageArrayNonUniformIndexing`), `shaderStorageImageExtendedFormats`, `depthClamp`, `drawIndirectCount`, optional `VK_EXT_mesh_shader`, and optional dedicated DMA transfer + async compute queue families.
     - Probe `vkGetPhysicalDeviceFormatProperties` at startup for `VK_FORMAT_R16G16_SNORM` (`COLOR_ATTACHMENT_BIT`, falling back to `R16G16_SFLOAT`) and `VK_FORMAT_B10G11R11_UFLOAT_PACK32` (`STORAGE_IMAGE_BIT`, falling back to `R32_UINT` bit-packed or `R16G16B16A16_SFLOAT`). Return typed `EngineError::Gpu(GpuError::UnsupportedAdapter)` when required Vulkan 1.3 ray-query or descriptor-indexing features are missing.
     - Auto-select runtime preset (`Radeon780M` at 30 FPS floor, `Radeon890M` at 60 FPS, `DiscreteDesktop` at 60+ FPS native / 100+ FPS upscaled, `HighEnd4K` at 90+ FPS) including per-preset VRAM budgets (`STATIC_GEOMETRY_ARENA_MB`, `BLAS_POOL_BUDGET_MB`, `TEXTURE_STREAM_BUDGET_MB`), `MAX_DYNAMIC_BLAS_COUNT`, `ALPHA_TEST_RT_RANGE_M`, `SHADOW_RESOLUTION_SCALE`, `MAX_VISIBLE_LIGHTS`, `GRID_X/Y/Z`, and `MAX_BINDLESS_TEXTURES`; expose `VisualStyle` (`Realistic` | `Comic`) and `UpscalerBackend` (`TaauRcas`).
  3. **Render Graph Barrier Tracker, `VkPipelineCache`, Memory Pools, Static Geometry Arena & Exact-Budget `BlasPool` (`crates/engine/src/gpu/`):**
     - Implement lightweight internal Render Graph state tracker emitting minimal `vkCmdPipelineBarrier2` layout/access transitions per pass and supporting transient scratch memory aliasing via `gpu-allocator`.
     - Implement on-disk `VkPipelineCache` load/save, triple-buffered dynamic pools, TLSF/free-list sub-allocated static geometry arena (retaining flat BLAS triangle-index slices + vertex tangent frames alongside meshlet buffers), exact-budget `BlasPool` sub-allocated inside a fixed `VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR` arena + scratch buffer (`vkGetAccelerationStructureBuildSizesKHR` + `VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR`), fixed Pass 1 upload staging ring (`STAGING_RING_MB`, DMA transfer queue capable), triple-buffered host-mapped mipmap-feedback readback ring, and N+3 timeline-semaphore deferred reclamation queue.
  4. **Shared ABI Expansion (`crates/shader_abi/src/lib.rs`, Scalar-Backed `#[repr(C)]` Only):**
     - Keep `crates/shader_abi` free of `glam` dependencies using scalar-backed `#[repr(C)]` types (`Vec4f`, `Mat4f`, `[f32; 4]`). Define `FrameConstants`, `PushConstants` (`<= 128 B`), `GpuLight` (`32 B`), `ClusterHeader` (`8 B`), `MeshletDesc`, `GpuVertex` (position, normal, tangent, uv), `GpuSubmeshDesc` (mapping `(instance_custom_index, geometry_index)` to static/dynamic BDA or pool-discriminated vertex/index slices and `MaterialId` for Pass 7 ray-hit shading), `GpuInstance` (including `EffectAttachment` parameters and submesh range), `GpuMaterial`, `HighlightCategory`, and compressed bone transform structs.
- **Verification Gate:**
  - **Software-safe CI:** `cargo clippy --workspace --all-targets -- -D warnings`, `shader_abi` byte-size/alignment tests, preset/arena/pool/N+3/barrier-tracker unit tests.
  - **Hardware-RT Vulkan 1.3 test (run with `VK_LAYER_KHRONOS_validation` enabled):** Headless Vulkan 1.3 validation of adapter probing, `RuntimeArray` `UPDATE_AFTER_BIND` descriptor sets, push constants, subgroups + barriers, `#[spirv(early_fragment_tests)]`, `VK_KHR_ray_query`, `VK_EXT_mesh_shader`, and `shaderStorageImageExtendedFormats` compute storage writes.

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
     - Implement independent residency policies for Physics (`PHYSICS_ACTIVE_RADIUS_M`), Geometry/LOD (`CAMERA_FAR_M`), Actors (`SIM_ACTIVE_RADIUS_M`), plus CPU-side residency state machines and unit tests for Ray-Tracing BLAS (`TLAS_ACTIVE_RADIUS_M + BLAS_PREFETCH_MARGIN_M`, wired to Pass 3 in Phase 4) and `BindlessTextureTable` mipmap feedback (`TEXTURE_STREAM_BUDGET_MB`, 16,384 to 65,536 `UPDATE_AFTER_BIND` slots, generational slots, O(1) `vkUpdateDescriptorSets` per promoted/demoted `VkImage`, atomic `min_lod` clamp, N+3 timeline deferred destroy, and host-mapped feedback drain wired to Pass 8 in Phase 3).
     - Implement player-gravity-only fallback when Tier-0 ground collider is not yet resident.
- **Verification Gate:**
  - Offline CLI compiles an animated multi-material glTF test scene and font into sector blobs (all meshlets `<= 64` verts and `<= 124` tris, 4+ LODs + HLOD, BC7/BC5 KTX2 including `2x2`/`1x1` mip tails, MSDF atlas, versioned Rapier chunks).
  - Async runtime streaming stress test confirming `0.00 ms` main-thread blocking disk I/O, clean ref-counted eviction under `TEXTURE_STREAM_BUDGET_MB`, and zero stale-request leaks.

### Phase 2: Retained Scene Boundary, Physics Refactor, Reverse-Z, Two-Phase Hi-Z, Thin Prepass & Far Sun Cascade (Passes 1, 4, 5)
- **Spec Sections:** 3, 4, 6, 10, 16.
- **Dependencies:** Phase 0, Phase 1.
- **Deliverables:**
  1. **Retained Scene & `RenderSnapshot` Evolution (`snapshot.rs`, `world.rs`, `save.rs`, `runner.rs`, `gpu/renderer.rs`):**
     - Replace per-frame 200 m CPU box collection (`RenderSnapshot::meshes` migration across `crates/engine` tests and `crates/game`) with retained GPU scene handles fed by an append-only monotonic delta queue fenced strictly by `delta.frame_index <= snapshot.frame_index` (accumulating mutations across `Sim` steps so `Parallel -> Sequential` overwrites and pre-`resumed` `GameApp::init` spawns never drop updates, while preventing concurrent `Sim N+1` mutations from bleeding into `Render N`) plus per-frame dynamic state (camera mode, `history_reset` invalidation flag, previous-frame transforms for motion vectors, dynamic lights, skeleton poses, highlight IDs, viewmodel state, `EffectAttachment` parameters, `VisualStyle`, `GizmoBox`). Drain fenced deltas and latch `history_reset` inside `VulkanRenderer::draw` before `acquire_next_image` early-returns so `OUT_OF_DATE`/`SUBOPTIMAL` frames never drop GPU scene state.
     - Implement sector actor hibernation/restoration via `ActorCheckpoint` (including skeletal animation state) into an in-memory Sector Delta Store.
  2. **Physics Refactor & Large-World `DVec3` Rebasing (`physics.rs`, `world.rs`, `snapshot.rs`, `save.rs`):**
     - Remove full-world `Octree::rebuild` on dynamic moves; back `SpatialQuery::query_octree` with Rapier's incremental broadphase/query pipeline and splice versioned pre-baked static sector colliders cleanly.
     - Migrate canonical CPU world coordinates across `ActorSpawnDesc`, `PhysicsCommand`, `WorldReadView`, `SpatialQuery`, `Aabb3`, `RaycastHit`, `CameraSnapshot`, `GizmoBox`, and `ActorCheckpoint` to `DVec3` with active physics-origin rebasing so `f32` `rapier3d` remains precise across large seamless worlds.
  3. **Pass 1 Camera-Relative Reverse-Z & Dual Camera:**
     - Reverse-Z `VK_FORMAT_D32_SFLOAT` (`1.0` near, `0.0` far, `GREATER_OR_EQUAL`, clear `0.0`, negative Vulkan viewport height), camera-relative `f32` rebasing, First-Person (`0.05 m` to `2,000 m`) and Isometric (`10 m` to `80 m`) modes, Halton jitter toggle (on for Realistic temporal upscaling, zero for Comic).
  4. **Passes 4 and 5 (Two-Phase Hi-Z Culling, Thin Attribute Prepass & Far Sun Cascade):**
     - Phase 1 compute meshlet culler (frustum, backface cone, screen-error LOD, prev-frame Hi-Z skipped on `history_reset`, subgroup compaction, solid vs. alpha-test vs. per-`EffectId` Pass-8 overlay split, isometric camera-to-player capsule/cone wall cutout routing with hysteresis).
     - Thin prepass (binding the global bindless texture set for alpha-test discard and BC5 normal maps) writing internal-resolution reverse-Z `D32_SFLOAT`, BC5 normal-mapped octahedral world normals (`R16G16_SNORM` or `R16G16_SFLOAT` fallback), roughness (`R8_UNORM`), rigid-body motion vectors (`R16G16_SFLOAT`; extended to skinned limb motion vectors in Phase 4), and `R16_UINT` Object ID + highlight category (`11 B/sample` total across 4 color attachments) via `VK_EXT_mesh_shader` or `vkCmdDrawIndexedIndirectCount` fallback.
     - Sun-frustum cull and `2048x2048` directional Far Sun Shadow Cascade raster pass (`depthClamp` enabled) for geometry beyond `TLAS_ACTIVE_RADIUS_M`.
     - Current-frame Hi-Z `min`-depth pyramid build (`R32_SFLOAT`), Phase 2 disoccluded meshlet pass, and merging Phase 1 + Phase 2 visible meshlets into Pass-8 replay indirect buffers.
- **Verification Gate:**
  - 10,000-meshlet behind-wall occlusion test (>95% hidden culled, zero popping on rapid camera rotation or `history_reset`).
  - Simultaneous `0.05 m` and `2,000 m` reverse-Z depth precision test, large-world `DVec3` origin-rebase test, and `RenderSchedule::Parallel` frame-fenced delta coherence test.

### Phase 3: Compute 3D Light Clustering, Dual-Style Opaque Forward Shading & Declarative GPU-Driven `RenderEffect` Hooks (Passes 6, 8)
- **Spec Sections:** 2, 3, 7, 8, 11, 16.
- **Dependencies:** Phase 2.
- **Deliverables:**
  1. **Pass 6 (Active-Cluster Depth Marking & Compute 3D Cluster Light Assignment):**
     - Exponential 3D cluster grid (`32` to `64` per axis, recomputed on resize/FOV/camera-mode change).
     - Lightweight depth-marking compute dispatch reading Pass 5 reverse-Z depth to flag occupied screen clusters before light assignment.
     - Subgroup ballot/shuffle/barrier light assignment for up to `4,096` dynamic `GpuLight` entries (`32 B` each) strictly over active occupied clusters.
  2. **Pass 8 (Opaque Clustered Forward Shading & Declarative Pluggable `RenderEffect` Hooks):**
     - Zero-overdraw `Depth == Equal` pass (`depth_write_enable = false`, `#[spirv(early_fragment_tests)]`) replaying the merged Pass 4+5 visible meshlet lists and writing `VK_FORMAT_B10G11R11_UFLOAT_PACK32` linear HDR color, sampling `RuntimeArray<Image!(...)>` bindless PBR textures and writing mipmap feedback requests.
     - Dual runtime-selectable shader entry points: **Realistic PBR** (cluster lights, Dynamic Time-of-Day Skybox IBL, built-in material Fresnel rim, Pass 7 denoised ray buffer hooks) and **Flat Comic** (quantized cel bands, flat ambient, disabled SH/specular reflections, optional Fresnel rim).
     - Extend declarative `EffectShaderSpec` (`crates/engine/src/effect.rs`) with typed GPU-driven hook stages (Pass 8 opaque material/overlay hooks reading per-`EffectId` visible meshlet queues + GPU instance `EffectAttachment` parameters via engine-provided mesh/vertex front-ends, Pass 10 transparent hooks, Pass 12 screen-space post hooks) while keeping `crates/fx_fresnel_rim` 100% free of `ash` and `unsafe`.
     - Migrate `crates/fx_fresnel_rim` to the GPU-driven Pass-8 `EffectShaderSpec` contract as the reference custom shader plugin crate.
- **Verification Gate:**
  - 4,096-light stress benchmark: Pass 6 within `0.10 to 0.40 ms` design reference on target desktop GPU, zero visible cluster boundary seams.
  - Runtime Realistic/Comic switch and declarative `crates/fx_fresnel_rim` custom `rust-gpu` hook verified on identical glTF assets with `VK_LAYER_KHRONOS_validation`.

### Phase 4: Skeletal Animation Evaluation, Compute Meshlet Skinning & Unified Hardware BVH (Passes 2, 3)
- **Spec Sections:** 9, 14.
- **Dependencies:** Phase 2.
- **Deliverables:**
  1. **Skeletal Animation Evaluation (`Sim` / `WorkerPool`):**
     - Sample glTF animation clips, blend poses, and compute hierarchical bone palettes into `RenderSnapshot` dynamic deltas; serialize clip ID, normalized playback time, and blend weights in `ActorCheckpoint`.
  2. **Pass 2 (Compute Meshlet Skinning + Skinned Motion Vectors):**
     - Upload compressed bone matrices to 12 MB triple-buffered storage pool.
     - Subgroup compute skinning writing Frame N deformed positions, normals, and tangents into the skinned vertex pool (`SHADER_DEVICE_ADDRESS | ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY`) while retaining previous-frame skinned positions (`36 to 72 MB` total) so Pass 4 emits accurate per-vertex motion vectors for animated limbs.
  3. **Pass 3 (In-Place Dynamic BLAS Refit, Async Static BLAS Compaction & Active-Radius TLAS):**
     - `VK_KHR_synchronization2` memory barrier from Pass 2 skinned vertex writes to acceleration structure build reads.
     - Execute true hardware in-place refit (`ALLOW_UPDATE | PREFER_FAST_TRACE`, `MODE_UPDATE`) on active dynamic character BLASes from Frame N skinned vertices (up to `MAX_DYNAMIC_BLAS_COUNT`).
     - Execute up to `STATIC_BLAS_BUILDS_PER_FRAME` queued static BLAS builds and drive the multi-frame async compaction state machine (`vkCmdWriteAccelerationStructuresPropertiesKHR` -> timeline retire -> `vkCmdCopyAccelerationStructureKHR COMPACT` -> atomic TLAS swap -> N+3 arena free) inside the exact `BLAS_POOL_BUDGET_MB` arena.
     - Rebuild single shared camera-relative TLAS for `RayReady` objects inside `TLAS_ACTIVE_RADIUS_M`, setting 24-bit `instanceCustomIndex` to each object's `GpuSubmeshDesc` range and setting `VkAccelerationStructureInstanceKHR::flags` (`FORCE_NO_OPAQUE` for near-range alpha-tested foliage inside `ALPHA_TEST_RT_RANGE_M`, `FORCE_OPAQUE` beyond `ALPHA_TEST_RT_RANGE_M`).
- **Verification Gate:**
  - Animated character ray-hit validation confirming `0`-frame lag between rasterized limbs and BVH hits (zero bind-pose spawn flash) and clean limb motion vectors in Pass 4.
  - 100-character timing benchmark on target desktop GPU verifying Pass 2 skinning, in-place dynamic BLAS refit (`0.15 to 0.35 ms` reference), async static BLAS compaction, and TLAS rebuild.

### Phase 5: Unified Hardware Ray Queries, Radiance Cascades GI & Dual-Mode Denoiser (Pass 7)
- **Spec Sections:** 4, 12, 17.
- **Dependencies:** Phase 3, Phase 4.
- **Deliverables:**
  1. **Stochastic Shadow Ray Queries + Far Sun Cascade Lookup:**
     - Automatic cluster-light importance sampling (brightness/distance, zero artist shadow flags).
     - 1 to 4 stochastic `TERMINATE_ON_FIRST_HIT` rays per pixel against shared TLAS (non-opaque candidate-intersection alpha-test inside `ALPHA_TEST_RT_RANGE_M`, opaque proxy via `FORCE_OPAQUE` beyond), blending seamlessly into Pass 4's `2048x2048` directional Far Sun Cascade beyond `TLAS_ACTIVE_RADIUS_M`.
  2. **Hit-Shaded Ray-Queried Glossy Reflections:**
     - 1 stochastic GGX ray per pixel where `roughness < REFLECTION_ROUGHNESS_CUTOFF` (0.5x res on iGPU floor, 1.0x on discrete, disabled in Comic). On hit, resolve `(instance_custom_index, geometry_index)` through `GpuSubmeshDesc` (BDA or static/dynamic pool discriminator) -> flat BLAS vertex/skinned slice + index slice + `MaterialId` -> bindless albedo/ORM textures and evaluate hit-point lighting. Zero SSAO/GTAO passes.
  3. **Hardware Ray-Queried Radiance Cascades:**
     - Screen-space depth-anchored Cascade 0 (`1.5 m` to `2.0 m` short rays) transitioning to coarse 3D camera-centered clipmap higher cascades against the shared TLAS (zero 3D SDFs), using `(instance_custom_index, geometry_index)` hit-point lookup + previous-frame merged radiance cache for leak-free multi-bounce GI.
  4. **Dual-Mode Denoiser:**
     - Temporal-spatial A-Trous filter using Pass 4 motion vectors and `history_reset` in Realistic mode (`24 MB` history at 1080p); spatial-only normal/depth-guided filter in Comic mode (`0 MB` history).
- **Verification Gate:**
  - Thin-wall cave/interior test verifying zero GI light leakage, true corner occlusion without SSAO, and accurate glossy hit shading on animated characters.
  - Offline QA profiler pass on target discrete desktop GPU confirming Shadows + Denoiser, Reflections, and Radiance Cascades stay within design reference budgets under `VK_LAYER_KHRONOS_validation`.

### Phase 6: Sky/Fog, Transparent GPU Radix Sort, Viewmodel, Pluggable Upscaling, Post-Process, Highlights & MSDF Text (Passes 9, 10, 11, 12)
- **Spec Sections:** 5, 13, 15, 16.
- **Dependencies:** Phase 3, Phase 5.
- **Deliverables:**
  1. **Pass 9 (Dynamic Time-of-Day Sky Atmosphere, Amortized IBL Re-Bake & Opaque Height Fog):** Physical sky LUT (`4 MB`) + amortized compute re-bake of the 16 MB Skybox IBL cubemap (prefiltered specular mips + SH diffuse irradiance) when sun angles change + opaque height fog in Realistic mode; stylized flat gradient sky (IBL re-bake disabled) in Comic mode where reverse-Z depth == `0.0`.
  2. **Pass 10 (Transparent Meshlet GPU Subgroup Radix Sort, Alpha Blend & Pluggable Overlay Hooks):** Camera-depth key generation, 3 to 4 dispatch back-to-front GPU subgroup radix sort (`1` to `4 MB`), single `vkCmdDrawMeshTasksIndirectCountEXT` / `vkCmdDrawIndexedIndirectCount` hardware alpha blend pass evaluating clustered lights, Dynamic Time-of-Day Skybox IBL, per-fragment height fog, and Pass 4 Far Sun Shadow Cascade lookup on cutout walls, plus Pass-10 `RenderEffect` hooks.
  3. **Pass 11 (First-Person Viewmodel Pass):** Dedicated near reverse-Z sub-range for player hands and weapons updating viewmodel octahedral normals and motion vectors with clustered/IBL lighting.
  4. **Pass 12 (Pure `rust-gpu` `UpscalerBackend`, Fragment Tonemapping, Post Hooks, Object ID Highlights, Debug Gizmos, MSDF Text & `UiOverlay`):**
     - **Unified `UpscalerBackend` (`TaauRcas`):** 100% `rust-gpu` temporal anti-aliasing upscaler with Halton jitter, `history_reset`, and RCAS sharpening writing target-resolution HDR, followed by a fullscreen AgX tonemap fragment pass writing directly into the sRGB swapchain image.
     - Comic mode: zero temporal history/jitter (`0.0 MB`), spatial scaling (EASU/bilinear + RCAS when `INTERNAL_RENDER_SCALE < 1.0`, or native `1.0` scale) + custom 3D color LUT + depth/octahedral-normal geometry ink outlines.
     - Registered Pass-12 `RenderEffect` post-process hooks, post-upscale `R16_UINT` Object ID edge detector + animated loot highlights, post-tonemap `GizmoBox` debug overlays, GPU MSDF text renderer (3D world signs + HUD hooks), and `UiOverlay::render` (`UiRenderTarget::composite_rgba8`).
- **Verification Gate:**
  - Isometric concave L-wall cutout depth-order and hysteresis test (zero manual artist mesh splitting).
  - First-person flush-wall weapon anti-clip and temporal-ghosting check.
  - Camera-mode switch (`FirstPerson <-> Isometric`), window resize, and style switch (`Realistic <-> Comic`) verifying clean `history_reset` with zero temporal ghosting.
  - MSDF crispness check at 720p, 1080p, 4K, and close-up 3D world signs.

### Phase 7: Sample Game Integration, Vulkan GPU Timestamp Profiler & Documentation Sync
- **Spec Sections:** All (1 through 18).
- **Dependencies:** Phase 6.
- **Deliverables:**
  1. **Dual-Camera, Dual-Style, Seamless-Streaming Sample (`crates/game`):**
     - Runtime camera mode switch (First-Person with viewmodel vs. Isometric with concave wall cutouts).
     - Runtime visual style switch (`Realistic` vs. `Comic`) and declarative `crates/fx_fresnel_rim` pluggable `RenderEffect` demonstration.
     - Multi-sector seamless traversal demonstrating async pop-in, animated glTF characters, dynamic torches/spells, glossy floors, alpha-tested foliage, glass/water, Object ID + Fresnel highlights, and 3D MSDF signs using pure reactive `GameApp` state.
  2. **12-Pass Vulkan GPU Timestamp Profiler (`crates/engine/src/gpu/profiler.rs`):**
     - Per-pass `VK_QUERY_TYPE_TIMESTAMP` (`vkCmdWriteTimestamp2`, scaled by `timestampPeriod`) across all 12 passes calibrated against the target discrete desktop GPU tier and Radeon 780M floor.
  3. **Documentation Sync (`docs/engine-architecture.md`):**
     - Align runtime diagram and notes with the completed 12-pass Vulkan 1.3 renderer, declarative `RenderEffect` hooks, and streaming subsystems.
- **Verification Gate:**
  - `cargo clippy --workspace --all-targets -- -D warnings`.
  - `cargo test --workspace` (software-safe suite) + hardware-RT Vulkan 1.3 integration suite with `VK_LAYER_KHRONOS_validation` + GPU-AV (zero validation errors).
  - Verified 12-pass GPU timestamp report calibrated on target discrete desktop hardware.

---

## 5. 5-Person Team Parallelization Map

After Phase 0 establishes Vulkan 1.3 capabilities, Render Graph barriers, GPU arenas, presets, and `shader_abi`, work runs in parallel across five ownership tracks:

| Engineer | Ownership | Phases |
| :--- | :--- | :--- |
| **Engineer 1 (GPU Core & Geometry)** | `gpu/` Render Graph Barriers & `VkPipelineCache`, Reverse-Z, Two-Phase Hi-Z, Meshlet Culling, Thin Prepass, Far Sun Cascade, Transparent Radix Sort, Viewmodel | 0, 2, 6 (Passes 10, 11) |
| **Engineer 2 (Ray Tracing & GI)** | Phase 0 `RayQuery` Spike & Exact-Budget `BlasPool` Arena, In-Place Dynamic Refit, Async Compaction, Active-Radius TLAS, Ray Queries (Shadows, Hit-Shaded Reflections), Radiance Cascades, Denoiser | 0, 4 (Pass 3), 5 (Pass 7) |
| **Engineer 3 (Shading, Lighting, Upscaling & Effects)** | Phase 0 `RuntimeArray`/Subgroup Spike & scalar-backed `shader_abi`, Active-Cluster Depth Mark & 3D Light Clustering, PBR/Comic Forward Shaders, Declarative GPU-Driven `RenderEffect` Hooks (`crates/fx_fresnel_rim`), Dynamic TOD Sky/IBL Re-Bake & Fog, Pure `rust-gpu` `TaauRcas`, AgX/Comic LUT, Object ID Highlights, Post-Tonemap Gizmos | 0, 3 (Passes 6, 8), 6 (Passes 9, 12) |
| **Engineer 4 (Assets, Streaming, Skinning & Text)** | Phase 0 Arena/Feedback Ring ABI, `crates/asset_pipeline`, `mmap` I/O Pool, Independent Residency Policies, `UPDATE_AFTER_BIND` `BindlessTextureTable` Mip Streamer, Skeletal Eval & Compute Skinning, MSDF Text | 0, 1, 4 (Pass 2), 6 (MSDF) |
| **Engineer 5 (Runtime, Physics, Game & QA)** | Phase 0 Device Startup Contract, Frame-Fenced Retained Scene / Pre-Acquire Delta Drain / `RenderSnapshot`, `physics.rs` Sector Colliders & `DVec3` Origin Rebasing, Sector Actor Hibernation, `crates/game` Showcase, 12-Pass Vulkan Timestamp Profiler & Validation Suite | 0, 2, 7 |

---

## 6. Phase Progress Tracker

| Phase | Scope | Status |
| :--- | :--- | :--- |
| **Baseline** | Private Vulkan 1.3 (`ash 0.38` + `ash-window 0.13` + `gpu-allocator 0.28`), `spirv-unknown-vulkan1.3`, declarative `RenderEffect`, private `UiRenderTarget::composite_rgba8`, typed `EngineError` | Completed |
| **Phase 0** | Pure `rust-gpu 0.11.0` Vulkan 1.3 RT/Mesh/Bindless spike, extended device/format probing, Render Graph barrier tracker, `VkPipelineCache`, static geometry & exact AS arenas, staging/readback rings, `shader_abi` | Not Started |
| **Phase 1** | `crates/asset_pipeline` (versioned colliders, `4x4`-padded BC7/BC5 KTX2, `rayon` + hash cache), `mmap` OS I/O pool, independent residency policies, O(1) `UPDATE_AFTER_BIND` bindless mip streamer | Not Started |
| **Phase 2** | Frame-fenced retained scene boundary, `physics.rs` refactor & full `DVec3` origin rebasing, Reverse-Z, Two-Phase Hi-Z, Thin Prepass (BC5 normal-mapped) & Far Sun Cascade (Passes 1, 4, 5) | Not Started |
| **Phase 3** | Compute 3D light clustering, dual-style opaque forward shading & declarative GPU-driven `RenderEffect` hooks (`crates/fx_fresnel_rim`) (Passes 6, 8) | Not Started |
| **Phase 4** | Skeletal eval, compute meshlet skinning (+ prev-frame positions) & Pass-3 in-place dynamic BLAS refit / async static compaction / TLAS pipeline (Passes 2, 3) | Not Started |
| **Phase 5** | Unified hardware ray queries (hit-shaded reflections, per-instance `FORCE_OPAQUE` foliage policy), Radiance Cascades GI & dual-mode denoiser (Pass 7) | Not Started |
| **Phase 6** | Dynamic TOD sky/IBL re-bake & fog, transparent radix sort (+ per-fragment lighting/fog), viewmodel (+ motion vectors/gizmos), pure `rust-gpu` `UpscalerBackend` (`TaauRcas`), fragment tonemap, highlights & MSDF text (Passes 9, 10, 11, 12) | Not Started |
| **Phase 7** | `crates/game` showcase, 12-pass Vulkan timestamp profiler (`vkCmdWriteTimestamp2`) & `docs/engine-architecture.md` sync | Not Started |
