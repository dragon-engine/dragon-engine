---
name: engine-architecture
description: Enforces the 9 core architectural invariants of crates/engine (type-enforced privacy, AoS/SoA physics coupling, dual executors, double-buffered RenderSchedule, dual clocks, lossless FIFO input, misuse-resistant async primitives, checkpoint reconstruction, decoupled UiOverlay, and rust-gpu shaders). Use when modifying crates/engine or reviewing engine-game boundaries.
---

# Engine Architecture Invariants

Apply this skill whenever you design, modify, or review modules inside `crates/engine` or change the `engine`/`game` public boundary.

## Core Mandate

`engine` cannot enforce Cargo, Clippy, or CI rules on external `game` crates. Enforce every safety and lifecycle invariant through private modules and Rust types inside `crates/engine`.

## Verification Checklist

1. **Privacy and Crate Independence:**
   - Keep `TaskSlab`, `TaskHandle`, `SlotMap`, `SoaStorage`, `rapier3d` sets, Vulkan 1.3 handles (`ash`, `ash-window`, `gpu-allocator`), and raw sync dependencies (`async-channel`, `event-listener`, `async-executor`) private inside `crates/engine`. Keep `RenderEffect` declarative (`EffectShaderSpec`) and `UiRenderTarget` free of raw GPU handles.
   - Keep zero `tokio` dependencies and zero game-specific names, assets, or types in `crates/engine`.
   - Keep `crates/engine/shaders/core_shader` self-contained with `[workspace]`, and direct `SpirvBuilder` output to `OUT_DIR`. Never add WGSL shaders.

2. **AoS/SoA Memory and Physics Coupling:**
   - Keep random-access entity data in private AoS `SlotMap<ActorId, ActorRecord>` accessed solely via synchronous closures (`read_actor`, `write_actor`).
   - Keep batch data (`positions`, `rotations`, `bounds`, `colliders`, `rigid_bodies`, `meshes`) in contiguous private `SoaStorage` synced with `Octree` and `rapier3d`.
   - Execute `Physics -> poll TaskSlab -> flush PhysicsCommand` in one simulation step, rebuilding `Octree` whenever `spawn_body`, `despawn_body`, or `SetPosition` mutates `SoaStorage`.

3. **Hybrid Dual-Executor and Four Runtime Rules:**
   - Run `TaskSlab` single-threaded with deterministic sorted-index polling and zero gameplay mutexes.
   - Never return raw `TaskHandle` to `game`; store handles inside `ActorRecord`, `ModeLayer`, or `WorldState`.
   - Enforce: `Option` on expired `ActorId`, drop `TaskHandle` on actor despawn, explicit `WaitTimeout` on waits, and cancel old buff `TaskHandle` on keyed refresh.
   - Offload heavy compute to `WorkerPool` (`async-executor`) using cloned snapshots (`Send + 'static`), requiring callers to revalidate `ActorId` after `.await`.

4. **Misuse-Resistant Async Primitives:**
   - Expose only FIFO `ActionSender<T>` / `ActionReceiver<T>` (never lossy `Notify`).
   - Require synchronous closures on `StateSignal::read`.
   - Require `WaitTimeout` (`WorldMs`, `UiMs`, or explicit `None`) on `ActionReceiver::next`, `StateSignal::wait_until`, and `ScriptContext::sleep`.

5. **RenderSchedule, Modalities, Input, and Checkpoints:**
   - Keep both `RenderSnapshot` buffers owned by `EngineRunner` for `Sequential` (`Low Latency`) and `Parallel` (`High Frame Rate`) schedules.
   - Gate `ClockDomain::World` via `ModeManager<FlatMode, StackMode>` while keeping `ClockDomain::Ui` at `1.0` and preserving `StackMode` across `FlatMode` toggles.
   - Route `winit` events to `UiOverlay::handle_event` first; batch unconsumed transitions losslessly and process sequentially against `ModeManager`.
   - Keep `SaveCodec` disk I/O strictly inside synchronous `GameApp::on_sync` while `ClockDomain::World` is paused.
