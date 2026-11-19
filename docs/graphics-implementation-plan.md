# Implementation Plan: Dual-Camera 3D RPG Graphics & Streaming Architecture

This document defines the phased implementation plan for realizing [Architectural Design Specifications for a Dual-Camera 3D RPG Graphics Engine](graphics-architecture.md) inside Dragon Engine while preserving all runtime invariants in [Engine Architecture Principles](engine-architecture.md).

---

## 1. Locked Architectural Decisions

These decisions are final and must not be reopened during implementation without an explicit Blueprint review:

1. **Linux/Vulkan-First Rollout:** Target native **Vulkan** (Linux first, plus Windows Vulkan) across Phases 0 to 5 using `rust-gpu` SPIR-V passthrough. Defer DX12 and Metal cross-backend support (offline SPIRV-Cross + DXC/MetalLib compilation emitting one DXIL module per entry point per https://github.com/gfx-rs/wgpu/issues/9991, defensive 1x1 dummy bindless fallback for adapters without `PARTIALLY_BOUND_BINDING_ARRAY`, Metal mesh-shader draw path) to Phase 7's dedicated cross-backend milestone.
2. **Strict 100% `rust-gpu`, Zero WGSL (with Pluggable Custom Shaders):** All engine-authored and game-authored compute, mesh, ray-query, graphics, and pluggable `RenderEffect` shaders live in `crates/engine/shaders/core_shader` and external `rust-gpu` crates (such as `crates/fx_fresnel_rim`) compiled into Cargo `OUT_DIR`. Use `spirv-std 0.11.0`'s native `RayQuery`, `GL_KHR_shader_subgroup`, `NonUniform`, and `TaskEXT`/`MeshEXT` APIs (inline SPIR-V `asm!` only if an edge-case intrinsic is absent). The only permitted non-`rust-gpu` binary exception is an optional feature-gated (`fsr-sdk`) official AMD FSR SDK 2.4.0 DLL on Windows (never GPL/patched override mods).
3. **Minimal Scoped `unsafe`:** Keep `unsafe_code = "deny"` in root `Cargo.toml`. Allow `#[allow(unsafe_code)]` strictly inside minimal private boundaries with no safe alternative:
   - `crates/engine/src/gpu/device.rs` (`wgpu::ExperimentalFeatures` and SPIR-V/passthrough shader creation),
   - `crates/engine/src/assets/mmap_store.rs` (read-only immutable `mmap` slices), and
   - `crates/engine/src/gpu/upscaler_fsr.rs` (optional feature-gated `fsr-sdk` `wgpu_hal` Vulkan/DX12 bridge for official AMD FSR SDK 2.4.0).
4. **Zero-Loading-Screen Async Pop-In (Frame-Pacing First):** Never stall simulation or rendering on disk I/O, decompression, GPU upload, BLAS build/compaction, or physics splicing. Accept visible asynchronous pop-in over frame-rate stutter.
5. **Independent Per-Subsystem Residency Policies:** Physics, Geometry/LOD, Ray-Tracing BLAS, Actors/AI, and GPU-Feedback Mipmap Streaming each own independent spatial residency policies and scaling parameters (no coupled monolithic sector rings).
6. **Lightweight Streaming Pacing (No Runtime Millisecond Timers):** Use only a fixed-size Pass 1 staging ring (`STAGING_RING_MB`), a runtime-preset per-frame static BLAS build cap (`STATIC_BLAS_BUILDS_PER_FRAME`), and explicit pool/VRAM budgets (`STATIC_GEOMETRY_ARENA_MB`, `BLAS_POOL_BUDGET_MB`, `TEXTURE_STREAM_BUDGET_MB`). Millisecond figures are design-time references and offline QA profiler targets (calibrated to the target discrete desktop GPU tier), never runtime timers.
7. **Zero-Lag Pass 2 / Pass 3 Ordering:** Execute **Pass 2 (Compute Meshlet Skinning)** before **Pass 3 (Dynamic BLAS Update/Rebuild, Async Static `wgpu::Blas` Compaction, and Active-Radius TLAS Rebuild)** so rasterization and hardware ray queries consume identical Frame N character geometry, while Pass 2 retains previous-frame skinned positions for Pass 4 motion vectors. Under `wgpu 30.0.1`, Pass 3 sets `ALLOW_UPDATE | PREFER_FAST_BUILD` and `PreferUpdate` (executing fast rebuilds until upstream refit lands: https://github.com/gfx-rs/wgpu/issues/6762).
8. **Zero Runtime PSO Stutter:** Precompile all fixed bindless compute, mesh, and render pipelines (including registered `RenderEffect` pipelines) once at startup.

---

## 2. Baseline Codebase Readiness Audit

Before Phase 0, the workspace (`wgpu 30.0.1`, `spirv-builder 0.11.0`, `spirv-std 0.11.0`, `glam 0.34.1`, `rapier3d 0.36.1`, `winit 0.30.13`, Rust edition `2024`) has these foundations and structural gaps:

### Ready Foundations
- **Simulation/Render Isolation (`crates/engine/src/runner.rs`, `snapshot.rs`):** `Renderer` is private and reads immutable double-buffered `RenderSnapshot<G>`. `RenderSchedule::Parallel` overlaps `Sim N+1` with `Render N`.
- **Shared ABI (`crates/shader_abi/src/lib.rs`):** `#![no_std]` crate with optional `bytemuck` derives shared by `engine` and `core_shader`.
- **Contiguous Transform Storage (`crates/engine/src/physics.rs`):** `SoaStorage` maintains dense `positions`, `rotations`, and `bounds` arrays.
- **Decoupled UI (`crates/engine/src/ui.rs`):** `UiOverlay::render` runs after the 3D scene pass via `UiRenderTarget`.

### Structural Gaps Addressed by Phases 0 to 3
1. **Immediate-Mode Box Snapshot vs. Retained GPU Scene (`world.rs`, `render_pipeline.rs`):** `collect_render_meshes` performs a hardcoded 200 m CPU octree query and clones box primitives every frame. Must transition to a retained GPU scene plus per-frame dynamic deltas (including previous-frame transforms for motion vectors and `history_reset` invalidation flags).
2. **Missing Asset Registry & Upload Queue (`runner.rs`, `world.rs`):** `SimRunner::new` runs `GameApp::init` before `wgpu::Device` exists. Requires a device-independent opaque handle registry bridged to an async GPU upload queue.
3. **Legacy Device Limits & SPIR-V Target (`renderer.rs`, `build.rs`):** `Renderer::new` requests `Features::empty()`, `downlevel_webgl2_defaults()`, and `ExperimentalFeatures::disabled()`. `build.rs` targets `spirv-unknown-vulkan1.1`.
4. **Missing Rendering Domain Primitives (`physics.rs`, `snapshot.rs`, `save.rs`):** Needs mesh/material/skeleton handles, dynamic lights, highlight categories, cutout flags, dual `CameraMode` (First-Person + viewmodel vs. Isometric + cutout target), `VisualStyle`, `UpscalerBackend`, and asset-ID + animation-state `ActorCheckpoint` schemas.
5. **Legacy CPU Vertex-Range `RenderEffect` Coupling (`effect.rs`, `crates/fx_fresnel_rim`):** Assumes CPU `SceneVertex` ranges (`EffectDrawItem { vertex_range }`). Keep pluggable custom `rust-gpu` shader support (`EffectRegistry`, `EngineRunner::run_with_effects`, `crates/fx_fresnel_rim`) while evolving `RenderEffect` to GPU-driven Pass-8, Pass-10, and Pass-12 hooks backed by `crates/shader_abi` and GPU instance parameters.
6. **Full-World `Octree::rebuild` (`physics.rs`):** Rebuilds the entire octree on every spawn, despawn, or `SetPosition`. Must separate versioned pre-baked static sector colliders from Rapier's incremental dynamic broadphase.
7. **Large-World Precision (`physics.rs`, `snapshot.rs`):** `rapier3d` and `glam::Vec3` use `f32`. Seamless large worlds require canonical `DVec3` (or sector-origin + local `f32`) coordinates with active-origin rebasing for `rapier3d` and camera-relative `f32` rebasing in Pass 1.

---

## 3. Target Workspace Structure

```text
dragon-engine/
├── crates/
│   ├── shader_abi/                  # Shared CPU/GPU #[repr(C)] ABI structs and constants (glam 0.34 parity)
│   ├── asset_pipeline/              # Offline CLI: glTF 2.0 -> meshlets/LODs/HLODs, BC7/BC5 KTX2, MSDF fonts, versioned Rapier collider chunks, sector manifests
│   ├── engine/
│   │   ├── build.rs                 # Multi-entry rust-gpu SPIR-V build (spirv-unknown-vulkan1.3) into OUT_DIR
│   │   ├── shaders/core_shader/     # 100% rust-gpu compute, mesh, ray-query, graphics, TAAU+RCAS, and MSDF shaders
│   │   └── src/
│   │       ├── gpu/                 # Device probe, scaling presets, triple-buffered pools, static geometry arena, budgeted Blas pool, staging/readback rings, parallel encoders, profiler, optional upscaler_fsr.rs
│   │       ├── assets/              # Opaque handle registry, private OS I/O thread pool, mmap store, ref-counted shared table, per-subsystem residency policies, immutable-swap bindless mip streamer
│   │       ├── render/              # 12-pass Render Graph modules (Passes 1 through 12, including UpscalerBackend and RenderEffect hook dispatch)
│   │       └── ...                  # Preserved core: clock, effect, executor, input, modes, physics, runner, save, snapshot, sync, ui, world
│   ├── fx_fresnel_rim/              # Reference pluggable custom rust-gpu RenderEffect crate (GPU-driven Pass 8/10/12 hook contract)
│   └── game/                        # Dual-camera, dual-style, seamless-streaming sample RPG
└── docs/
    ├── engine-architecture.md       # Runtime invariants
    ├── graphics-architecture.md     # Graphics and streaming Master Blueprint
    └── graphics-implementation-plan.md
```

---

## 4. Phased Implementation Plan

### Phase 0: Toolchain Verification, Native `wgpu` Capabilities, GPU Arenas, and `shader_abi`
- **Spec Sections:** 1, 7, 8, 18.
- **Dependencies:** None.
- **Deliverables:**
  1. **Pure `rust-gpu 0.11.0` Vulkan Capability Spike & Build Upgrade (`crates/engine/build.rs`, `core_shader`):**
     - Upgrade `SpirvBuilder` target to `spirv-unknown-vulkan1.3` with required SPIR-V capabilities.
     - Verify pure `rust-gpu` compilation and `wgpu 30.0.1` Vulkan pipeline creation using `spirv-std 0.11.0`'s native APIs for subgroups (`GL_KHR_shader_subgroup`), fixed-size `[Image!(...); N]` `NonUniform` bindless texture arrays (not unsized `RuntimeArray`, as `wgpu-hal 30.0.1` does not enable Vulkan `runtime_descriptor_array`: https://github.com/gfx-rs/wgpu/issues/7347, https://github.com/gfx-rs/wgpu/issues/3637), `wgpu` Immediates (`Features::IMMEDIATES`, `<= 128 B`), `RayQuery` (including non-opaque candidate-intersection alpha-test traversal), and `TaskEXT`/`MeshEXT` mesh shaders via `PASSTHROUGH_SHADERS`.
     - Benchmark `Device::create_bind_group` CPU cost at `2,048` and `4,096` bindless texture slots to size coalesced rebuild pacing (https://github.com/gfx-rs/wgpu/issues/8557).
  2. **Device Capability Probe & Dynamic Runtime Presets (`crates/engine/src/gpu/device.rs`, `config.rs`):**
     - Isolate `#[allow(unsafe_code)]` for `wgpu::ExperimentalFeatures` and SPIR-V passthrough creation inside `gpu/device.rs`; return a typed startup error from `EngineRunner::run` when a host lacks Vulkan 1.3 or `EXPERIMENTAL_RAY_QUERY`.
     - Request `TEXTURE_COMPRESSION_BC`, `DEPTH_CLIP_CONTROL`, `TEXTURE_BINDING_ARRAY`, `SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING`, `PARTIALLY_BOUND_BINDING_ARRAY` (where supported), `IMMEDIATES`, `MULTI_DRAW_INDIRECT_COUNT`, `SUBGROUP`, `EXPERIMENTAL_MESH_SHADER`, `EXPERIMENTAL_RAY_QUERY`, `PASSTHROUGH_SHADERS`, and `TIMESTAMP_QUERY`, alongside `Limits::using_minimum_supported_acceleration_structure_values()`, `using_recommended_minimum_mesh_shader_values()`, `max_immediate_size = 128`, and clamped `max_binding_array_elements_per_shader_stage`.
     - Auto-select runtime preset (`Radeon780M` at 30 FPS floor, `Radeon890M` at 60 FPS, `DiscreteDesktop` at 100+ FPS, `HighEnd4K` at 90+ FPS) including per-preset VRAM budgets (`STATIC_GEOMETRY_ARENA_MB`, `BLAS_POOL_BUDGET_MB`, `TEXTURE_STREAM_BUDGET_MB`) and `MAX_DYNAMIC_BLAS_COUNT`; expose `VisualStyle` (`Realistic` | `Comic`) and `UpscalerBackend` (`TaauRcas` default).
  3. **Memory Pools, Static Geometry Arena, Budgeted `BlasPool` & Parallel Encoders (`crates/engine/src/gpu/`):**
     - Implement triple-buffered dynamic pools, TLSF/free-list sub-allocated static geometry arena (retaining flat BLAS triangle-index slices alongside meshlet buffers), private `BlasPool` tracker using a deterministic CPU triangle-count byte estimator (`wgpu::Blas` hides byte size: https://github.com/gfx-rs/wgpu/issues/6762, https://github.com/gfx-rs/wgpu/pull/7285; uses safe `Device::create_blas` / `Queue::compact_blas`), fixed Pass 1 upload staging ring (`STAGING_RING_MB`), triple-buffered `MAP_READ` mipmap-feedback staging ring, and N+3 deferred reclamation queue.
     - Implement worker-thread `CommandEncoder` recording pool scoped to independent render-graph stages (Pass 3 BVH, Pass 4 Far Sun Cascade, Pass 4 Thin Prepass) with explicit disjoint thread-pool roles under `RenderSchedule::Parallel`.
  4. **Shared ABI Expansion (`crates/shader_abi/src/lib.rs`):**
     - Define `FrameConstants`, `ImmediateConstants` (`<= 128 B`), `GpuLight` (`32 B`), `ClusterHeader` (`8 B`), `MeshletDesc`, `GpuSubmeshDesc` (mapping `(instance_custom_index, geometry_index)` to flat BLAS vertex/index slices and `MaterialId` for Pass 7 ray-hit shading), `GpuInstance` (including `EffectAttachment` parameters and submesh range), `GpuMaterial`, `HighlightCategory`, and compressed bone transform structs.
- **Verification Gate:**
  - **Software-safe CI:** `cargo clippy --workspace --all-targets -- -D warnings`, `shader_abi` byte-size/alignment tests, preset/arena/pool/N+3 unit tests.
  - **Hardware-RT Vulkan test (tagged `#[ignore]` or feature-gated for non-RT CI):** Headless `wgpu` Vulkan validation of adapter probing, `IMMEDIATES`, `SUBGROUP`, `EXPERIMENTAL_RAY_QUERY`, `EXPERIMENTAL_MESH_SHADER`, and `rust-gpu` SPIR-V passthrough pipeline creation.

### Phase 1: Offline Asset Pipeline (`crates/asset_pipeline`) & Async Streaming Runtime
- **Spec Sections:** 10, 11, 15, 18.
- **Dependencies:** Phase 0 (`shader_abi`, GPU arenas, staging/readback rings).
- **Deliverables:**
  1. **Offline Asset Compiler (`crates/asset_pipeline`):**
     - Ingest glTF 2.0 (meshes, PBR materials, alpha-test flags, skeletons, animations, scene transforms).
     - Generate 64-vertex / 124-triangle meshlets with bounding spheres and backface cones, 4 to 6 automatic LOD levels (`meshoptimizer`), and merged per-sector HLOD proxies.
     - Compress albedo/ORM textures to BC7 and normal maps to BC5 inside KTX2 containers.
     - Generate MSDF font atlases and glyph metric tables.
     - Pre-bake versioned serialized `rapier3d` static collider chunks (`rapier_schema_version` + raw triangle-soup/convex fallback) separating Tier-0 ground/walkmesh colliders from detailed prop colliders.
     - Emit sector containers and a reference-counted shared asset manifest.
  2. **Runtime Memory-Mapped Store & Private OS I/O Pool (`crates/engine/src/assets/mmap_store.rs`, `io_pool.rs`):**
     - Isolate `#[allow(unsafe_code)]` for read-only `mmap` inside `assets/mmap_store.rs`.
     - Run all `mmap` page reads, KTX2 mip reads, and collider deserialization on a private OS I/O thread pool (zero I/O in `TaskSlab` or render thread) with instant stale-request cancellation and lock-free handoff compatible with `RenderSchedule::Parallel`.
  3. **Device-Independent Handle Registry & Independent Residency Coordinators (`crates/engine/src/assets/`):**
     - Expose opaque handles (`MeshHandle`, `MaterialHandle`, `TextureHandle`, `SkeletonHandle`, `FontHandle`).
     - Implement independent residency policies for Physics (`PHYSICS_ACTIVE_RADIUS_M`), Geometry/LOD (`CAMERA_FAR_M`), Actors (`SIM_ACTIVE_RADIUS_M`), plus CPU-side residency state machines and unit tests for Ray-Tracing BLAS (`TLAS_ACTIVE_RADIUS_M + BLAS_PREFETCH_MARGIN_M`, wired to Pass 3 in Phase 4) and `BindlessTextureTable` mipmap feedback (`TEXTURE_STREAM_BUDGET_MB`, `2,048` to `4,096` default live slots, generational slots, immutable mip-chain `wgpu::Texture` swap, coalesced bindless `BindGroup` rebuild, atomic `min_lod` clamp, N+3 deferred texture drop, and `MAP_READ` feedback drain wired to Pass 8 in Phase 3; https://github.com/gfx-rs/wgpu/issues/8557).
     - Implement player-gravity-only fallback when Tier-0 ground collider is not yet resident.
- **Verification Gate:**
  - Offline CLI compiles an animated multi-material glTF test scene and font into sector blobs (all meshlets `<= 64` verts and `<= 124` tris, 4+ LODs + HLOD, BC7/BC5 KTX2, MSDF atlas, versioned Rapier chunks).
  - Async runtime streaming stress test confirming `0.00 ms` main-thread blocking disk I/O, clean ref-counted eviction under `TEXTURE_STREAM_BUDGET_MB`, and zero stale-request leaks.

### Phase 2: Retained Scene Boundary, Physics Refactor, Reverse-Z, Two-Phase Hi-Z, Thin Prepass & Far Sun Cascade (Passes 1, 4, 5)
- **Spec Sections:** 3, 4, 6, 10, 16.
- **Dependencies:** Phase 0, Phase 1.
- **Deliverables:**
  1. **Retained Scene & `RenderSnapshot` Evolution (`snapshot.rs`, `world.rs`, `save.rs`, `renderer.rs`):**
     - Replace per-frame 200 m CPU box collection with retained GPU scene handles plus per-frame dynamic deltas (camera mode, `history_reset` invalidation flag, previous-frame transforms for motion vectors, dynamic lights, skeleton poses, highlight IDs, viewmodel state, `EffectAttachment` parameters, `VisualStyle`, `GizmoBox`). Drain retained-scene deltas and latch `history_reset` inside `Renderer::draw` before `surface.get_current_texture()` early returns so `Occluded`/`Outdated`/`Suboptimal`/`Lost` frames never drop GPU scene state.
     - Implement sector actor hibernation/restoration via `ActorCheckpoint` (including skeletal animation state) into an in-memory Sector Delta Store.
  2. **Physics Refactor & Large-World Rebasing (`physics.rs`, `world.rs`, `save.rs`):**
     - Remove full-world `Octree::rebuild` on dynamic moves; back `SpatialQuery::query_octree` with Rapier's incremental broadphase/query pipeline and splice versioned pre-baked static sector colliders cleanly.
     - Migrate canonical CPU world coordinates (`ActorSpawnDesc`, `PhysicsCommand`, `WorldReadView`, `CameraSnapshot`, `ActorCheckpoint`) to `DVec3` (or sector-origin + local `f32`) with active physics-origin rebasing so `f32` `rapier3d` remains precise across large seamless worlds.
  3. **Pass 1 Camera-Relative Reverse-Z & Dual Camera:**
     - Reverse-Z `Depth32Float` (`1.0` near, `0.0` far, `GreaterEqual`, clear `0.0`), camera-relative `f32` rebasing, First-Person (`0.05 m` to `2,000 m`) and Isometric (`10 m` to `80 m`) modes, Halton jitter toggle (on for Realistic temporal upscaling, zero for Comic).
  4. **Passes 4 and 5 (Two-Phase Hi-Z Culling, Thin Attribute Prepass & Far Sun Cascade):**
     - Phase 1 compute meshlet culler (frustum, backface cone, screen-error LOD, prev-frame Hi-Z skipped on `history_reset`, subgroup compaction, solid vs. alpha-test split, isometric camera-to-player capsule/cone wall cutout routing with hysteresis).
     - Thin prepass writing internal-resolution reverse-Z `Depth32Float`, octahedral normals, roughness, rigid-body motion vectors (`Rg16Float`; extended to skinned limb motion vectors in Phase 4), and `R16Uint` Object ID + highlight category via native mesh shaders or `MultiDrawIndirectCount` fallback (with separate alpha-test prepass draw list).
     - Sun-frustum cull and `2048x2048` directional Far Sun Shadow Cascade raster pass for geometry beyond `TLAS_ACTIVE_RADIUS_M`.
     - Current-frame Hi-Z `min`-depth pyramid build and Phase 2 disoccluded meshlet pass.
- **Verification Gate:**
  - 10,000-meshlet behind-wall occlusion test (>95% hidden culled, zero popping on rapid camera rotation or `history_reset`).
  - Simultaneous `0.05 m` and `2,000 m` reverse-Z depth precision test and large-world origin-rebase test.

### Phase 3: Compute 3D Light Clustering, Dual-Style Opaque Forward Shading & Pluggable `RenderEffect` Hooks (Passes 6, 8)
- **Spec Sections:** 2, 3, 7, 8, 11, 16.
- **Dependencies:** Phase 2.
- **Deliverables:**
  1. **Pass 6 (Compute 3D Cluster Light Assignment):**
     - Exponential 3D cluster grid (`32` to `64` per axis, recomputed on resize/FOV/camera-mode change).
     - Subgroup ballot/shuffle light assignment for up to `4,096` dynamic `GpuLight` entries (`32 B` each).
  2. **Pass 8 (Opaque Clustered Forward Shading & Pluggable Custom `rust-gpu` Hooks):**
     - Zero-overdraw `Depth == Equal` pass (`depth_write_enabled = false`) sampling bindless PBR textures and writing mipmap feedback requests.
     - Dual runtime-selectable shader entry points: **Realistic PBR** (cluster lights, Skybox IBL, built-in material Fresnel rim, Pass 7 denoised ray buffer hooks) and **Flat Comic** (quantized cel bands, flat ambient, disabled SH/specular reflections, optional Fresnel rim).
     - Evolve `crates/engine/src/effect.rs` (`RenderEffect`, `EffectRegistry`, `EngineRunner::run_with_effects`) from legacy CPU `SceneVertex` ranges to typed GPU-driven `rust-gpu` hook stages (Pass 8 opaque material/overlay hooks, Pass 10 transparent hooks, Pass 12 post-process hooks) reading `@group(0)` engine bindings and GPU instance `EffectAttachment` parameters, adding an engine-mediated shader-loading helper on `EffectInitContext` (backed by `gpu/device.rs`) so external effect crates load passthrough SPIR-V without `unsafe`.
     - Migrate `crates/fx_fresnel_rim` to the new GPU-driven `RenderEffect` hook contract as the reference custom shader plugin crate.
- **Verification Gate:**
  - 4,096-light stress benchmark: Pass 6 within `0.10 to 0.40 ms` design reference on target desktop GPU, zero visible cluster boundary seams.
  - Runtime Realistic/Comic switch and pluggable `crates/fx_fresnel_rim` custom `rust-gpu` hook verified on identical glTF assets.

### Phase 4: Skeletal Animation Evaluation, Compute Meshlet Skinning & Unified Hardware BVH (Passes 2, 3)
- **Spec Sections:** 9, 14.
- **Dependencies:** Phase 2.
- **Deliverables:**
  1. **Skeletal Animation Evaluation (`Sim` / `WorkerPool`):**
     - Sample glTF animation clips, blend poses, and compute hierarchical bone palettes into `RenderSnapshot` dynamic deltas; serialize clip ID, normalized playback time, and blend weights in `ActorCheckpoint`.
  2. **Pass 2 (Compute Meshlet Skinning + Skinned Motion Vectors):**
     - Upload compressed bone matrices to 12 MB triple-buffered storage pool.
     - Subgroup compute skinning writing Frame N deformed positions and normals into the skinned vertex pool while retaining previous-frame skinned positions (`36 to 72 MB` total) so Pass 4 emits accurate per-vertex motion vectors for animated limbs.
  3. **Pass 3 (Dynamic BLAS Update/Rebuild, Async Static `wgpu::Blas` Compaction & Active-Radius TLAS):**
     - GPU memory barrier from Pass 2 skinned vertex writes to acceleration structure reads.
     - Update active dynamic character `wgpu::Blas` handles (`ALLOW_UPDATE | PREFER_FAST_BUILD`, `PreferUpdate`) from Frame N skinned vertices (bounded by `MAX_DYNAMIC_BLAS_COUNT` while `wgpu 30.0.1` executes fast full rebuilds instead of refit: https://github.com/gfx-rs/wgpu/issues/6762).
     - Execute up to `STATIC_BLAS_BUILDS_PER_FRAME` queued static BLAS builds and drive the multi-frame async compaction state machine (`Blas::prepare_compaction_async` -> poll `ready_for_compaction()` -> `Queue::compact_blas` -> atomic TLAS swap -> N+3 drop of uncompacted `Blas`) within `BLAS_POOL_BUDGET_MB` (CPU-estimated inside `BlasPool`).
     - Rebuild single shared camera-relative TLAS for `RayReady` objects inside `TLAS_ACTIVE_RADIUS_M`, setting 24-bit `instance_custom_index` to each object's `GpuInstance` / submesh table index.
- **Verification Gate:**
  - Animated character ray-hit validation confirming `0`-frame lag between rasterized limbs and BVH hits (zero bind-pose spawn flash) and clean limb motion vectors in Pass 4.
  - 100-character timing benchmark on target desktop GPU verifying Pass 2 skinning, dynamic BLAS refit, async static BLAS compaction, and TLAS rebuild.

### Phase 5: Unified Hardware Ray Queries, Radiance Cascades GI & Dual-Mode Denoiser (Pass 7)
- **Spec Sections:** 4, 12, 17.
- **Dependencies:** Phase 3, Phase 4.
- **Deliverables:**
  1. **Stochastic Shadow Ray Queries + Far Sun Cascade Lookup:**
     - Automatic cluster-light importance sampling (brightness/distance, zero artist shadow flags).
     - 1 to 4 stochastic `TERMINATE_ON_FIRST_HIT` rays per pixel against shared TLAS (with non-opaque any-hit alpha test for near-range foliage), blending seamlessly into Pass 4's `2048x2048` directional Far Sun Cascade beyond `TLAS_ACTIVE_RADIUS_M`.
  2. **Hit-Shaded Ray-Queried Glossy Reflections:**
     - 1 stochastic GGX ray per pixel where `roughness < REFLECTION_ROUGHNESS_CUTOFF` (0.5x res on iGPU floor, 1.0x on discrete, disabled in Comic). On hit, resolve `(instance_custom_index, geometry_index)` through `GpuSubmeshDesc` -> flat BLAS geometry/skinned vertex slice + index slice + `MaterialId` -> bindless albedo/ORM textures and evaluate hit-point lighting. Zero SSAO/GTAO passes.
  3. **Hardware Ray-Queried Radiance Cascades:**
     - Screen-space depth-anchored Cascade 0 (`1.5 m` to `2.0 m` short rays) transitioning to coarse 3D camera-centered clipmap higher cascades against the shared TLAS (zero 3D SDFs), using `(instance_custom_index, geometry_index)` hit-point lookup + previous-frame merged radiance cache for leak-free multi-bounce GI.
  4. **Dual-Mode Denoiser:**
     - Temporal-spatial A-Trous filter using Pass 4 motion vectors and `history_reset` in Realistic mode (`24 MB` history at 1080p); spatial-only normal/depth-guided filter in Comic mode (`0 MB` history).
- **Verification Gate:**
  - Thin-wall cave/interior test verifying zero GI light leakage, true corner occlusion without SSAO, and accurate glossy hit shading on animated characters.
  - Offline QA profiler pass on target discrete desktop GPU confirming Shadows + Denoiser, Reflections, and Radiance Cascades stay within design reference budgets.

### Phase 6: Sky/Fog, Transparent GPU Radix Sort, Viewmodel, Pluggable Upscaling, Post-Process, Highlights & MSDF Text (Passes 9, 10, 11, 12)
- **Spec Sections:** 5, 13, 15, 16.
- **Dependencies:** Phase 3, Phase 5.
- **Deliverables:**
  1. **Pass 9 (Sky Atmosphere & Opaque Height Fog):** Physical sky LUT (`4 MB`) + opaque height fog in Realistic mode; stylized flat gradient sky in Comic mode where reverse-Z depth == `0.0`.
  2. **Pass 10 (Transparent Meshlet GPU Subgroup Radix Sort, Alpha Blend & Pluggable Overlay Hooks):** Camera-depth key generation, 3 to 4 dispatch back-to-front GPU subgroup radix sort (`1` to `4 MB`), single mesh-shader / `MultiDrawIndirectCount` hardware alpha blend pass evaluating clustered lights, skybox IBL, per-fragment height fog, and Pass 4 Far Sun Shadow Cascade lookup on cutout walls, plus Pass-10 `RenderEffect` hooks.
  3. **Pass 11 (First-Person Viewmodel Pass):** Dedicated near reverse-Z sub-range for player hands and weapons updating viewmodel octahedral normals and motion vectors with clustered/IBL lighting.
  4. **Pass 12 (Pluggable `UpscalerBackend`, Tonemapping, Post Hooks, Object ID Highlights, Debug Gizmos, MSDF Text & `UiOverlay`):**
     - **Default Linux-first `UpscalerBackend` (`TaauRcas`):** 100% `rust-gpu` temporal anti-aliasing upscaler with Halton jitter, `history_reset`, and RCAS sharpening + AgX tonemapping.
     - **Optional feature-gated `fsr-sdk` backend (`crates/engine/src/gpu/upscaler_fsr.rs`, Windows Vulkan/DX12):** Official AMD FSR SDK 2.4.0 (`FSR 4.1.2` with automatic Lite INT8 model selection + `FSR 3.1.6` fallback) via isolated `unsafe` `wgpu_hal` interop.
     - Comic mode: zero temporal history/jitter (`0.0 MB`), spatial scaling (EASU/bilinear + RCAS when `INTERNAL_RENDER_SCALE < 1.0`, or native `1.0` scale) + custom 3D color LUT + depth/octahedral-normal geometry ink outlines.
     - Registered Pass-12 `RenderEffect` post-process hooks, post-upscale `R16Uint` Object ID edge detector + animated loot highlights, post-tonemap `GizmoBox` debug overlays, GPU MSDF text renderer (3D world signs + HUD hooks), and `UiOverlay::render`.
- **Verification Gate:**
  - Isometric concave L-wall cutout depth-order and hysteresis test (zero manual artist mesh splitting).
  - First-person flush-wall weapon anti-clip and temporal-ghosting check.
  - Camera-mode switch (`FirstPerson <-> Isometric`) and style switch (`Realistic <-> Comic`) verifying clean `history_reset` with zero temporal ghosting.
  - MSDF crispness check at 720p, 1080p, 4K, and close-up 3D world signs.

### Phase 7: Sample Game Integration, GPU Profiler, Cross-Backend Milestone & Documentation Sync
- **Spec Sections:** All (1 through 18).
- **Dependencies:** Phase 6.
- **Deliverables:**
  1. **Dual-Camera, Dual-Style, Seamless-Streaming Sample (`crates/game`):**
     - Runtime camera mode switch (First-Person with viewmodel vs. Isometric with concave wall cutouts).
     - Runtime visual style switch (`Realistic` vs. `Comic`) and custom `crates/fx_fresnel_rim` pluggable `RenderEffect` demonstration.
     - Multi-sector seamless traversal demonstrating async pop-in, animated glTF characters, dynamic torches/spells, glossy floors, alpha-tested foliage, glass/water, Object ID + Fresnel highlights, and 3D MSDF signs using pure reactive `GameApp` state.
  2. **12-Pass GPU Timestamp Profiler (`crates/engine/src/gpu/profiler.rs`):**
     - Pass-level `wgpu::QuerySet` `timestamp_writes` (`Features::TIMESTAMP_QUERY`, avoiding Apple Silicon encoder-timestamp restrictions) across all 12 passes calibrated against the target discrete desktop GPU tier.
  3. **Cross-Backend DX12 & Metal Milestone:**
     - Add offline SPIRV-Cross + DXC/MetalLib cross-compilation for `core_shader` and external `RenderEffect` crates with `PassthroughShaderEntryPoint` reflection, emitting one DXIL module per entry point to satisfy `wgpu-core 30.0.1`'s DXIL single-entry constraint (https://github.com/gfx-rs/wgpu/issues/9991).
     - Add Metal-specific accommodations (native mesh-shader draw path without `MULTI_DRAW_INDIRECT_COUNT`, plus defensive 1x1 dummy bindless fallback views if an adapter lacks `PARTIALLY_BOUND_BINDING_ARRAY`).
  4. **Documentation Sync (`docs/engine-architecture.md`):**
     - Align runtime diagram and notes with the completed 12-pass renderer, pluggable `RenderEffect` hooks, and streaming subsystems.
- **Verification Gate:**
  - `cargo clippy --workspace --all-targets -- -D warnings`.
  - `cargo test --workspace` (software-safe suite) + hardware-RT Vulkan integration suite.
  - Verified 12-pass GPU timestamp report calibrated on target discrete desktop hardware.

---

## 5. 5-Person Team Parallelization Map

After Phase 0 establishes GPU arenas, presets, and `shader_abi`, work runs in parallel across five ownership tracks:

| Engineer | Ownership | Phases |
| :--- | :--- | :--- |
| **Engineer 1 (GPU Core & Geometry)** | `gpu/`, Reverse-Z, Two-Phase Hi-Z, Meshlet Culling, Thin Prepass, Far Sun Cascade, Transparent Radix Sort, Viewmodel | 0, 2, 6 (Passes 10, 11) |
| **Engineer 2 (Ray Tracing & GI)** | Phase 0 `RayQuery` Spike & `BlasPool` ABI, Async Compaction, Active-Radius TLAS, Ray Queries (Shadows, Hit-Shaded Reflections), Radiance Cascades, Denoiser | 0, 4 (Pass 3), 5 (Pass 7) |
| **Engineer 3 (Shading, Lighting, Upscaling & Effects)** | Phase 0 Bindless/Subgroup Spike & `shader_abi`, 3D Light Clustering, PBR/Comic Forward Shaders, GPU-Driven `RenderEffect` Hooks (`crates/fx_fresnel_rim`), Sky/Fog, `rust-gpu` TAAU+RCAS & Optional `fsr-sdk` Bridge, AgX/Comic LUT, Object ID Highlights, Post-Tonemap Gizmos | 0, 3 (Passes 6, 8), 6 (Passes 9, 12) |
| **Engineer 4 (Assets, Streaming, Skinning & Text)** | Phase 0 Arena/Feedback Ring ABI, `crates/asset_pipeline`, `mmap` I/O Pool, Independent Residency Policies, `BindlessTextureTable` Mip Streamer, Skeletal Eval & Compute Skinning, MSDF Text | 0, 1, 4 (Pass 2), 6 (MSDF) |
| **Engineer 5 (Runtime, Physics, Game, Cross-Backend & QA)** | Phase 0 Device Startup Contract, Retained Scene / Pre-Swapchain Delta Drain / `RenderSnapshot`, `physics.rs` Sector Colliders & `DVec3` Origin Rebasing, Sector Actor Hibernation, `crates/game` Showcase, 12-Pass Profiler, DX12/Metal Cross-Backend Milestone | 0, 2, 7 |

---

## 6. Phase Progress Tracker

| Phase | Scope | Status |
| :--- | :--- | :--- |
| **Phase 0** | Pure `rust-gpu 0.11.0` Vulkan spike, native `wgpu 30.0.1` presets, `IMMEDIATES`, static geometry arena, budgeted `Blas` pool, staging/readback rings, `shader_abi` | Not Started |
| **Phase 1** | `crates/asset_pipeline` (versioned colliders), `mmap` OS I/O pool, independent residency policies, immutable-swap bindless mip streamer | Not Started |
| **Phase 2** | Retained scene boundary, `physics.rs` refactor & origin rebasing, Reverse-Z, Two-Phase Hi-Z, Thin Prepass & Far Sun Cascade (Passes 1, 4, 5) | Not Started |
| **Phase 3** | Compute 3D light clustering, dual-style opaque forward shading & GPU-driven pluggable `RenderEffect` hooks (`crates/fx_fresnel_rim`) (Passes 6, 8) | Not Started |
| **Phase 4** | Skeletal eval, compute meshlet skinning (+ prev-frame positions) & Pass-3 async `wgpu::Blas` compaction / TLAS pipeline (Passes 2, 3) | Not Started |
| **Phase 5** | Unified hardware ray queries (hit-shaded reflections, alpha-test policy), Radiance Cascades GI & dual-mode denoiser (Pass 7) | Not Started |
| **Phase 6** | Sky/fog, transparent radix sort (+ per-fragment lighting/fog), viewmodel (+ motion vectors/gizmos), `UpscalerBackend` (`TaauRcas` + optional `fsr-sdk`), highlights & MSDF text (Passes 9, 10, 11, 12) | Not Started |
| **Phase 7** | `crates/game` showcase, pass-level 12-pass GPU timestamp profiler, DX12/Metal cross-backend milestone & `docs/engine-architecture.md` sync | Not Started |
