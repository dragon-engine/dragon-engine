# Project Agent Rules: Dragon Engine

## 1. Core Principle: Reactive Architecture and Pure State

All gameplay logic in `crates/game` must be reactive and event-driven. State must represent only domain data, never transient input bookkeeping or polling flags.

### Mandatory Rules

1. **State Is Pure Data Only:**
   - `GameApp` structs, AoS `ActorData`, and `StateSignal<T>` values must store only domain state (for example: entity IDs, HP, potions, quest stage, camera angle, active buffs).
   - Do not store input-tracking booleans (such as `move_w`, `is_attack_pressed`, or `key_held`) on `GameApp` or `ActorData`.
   - Do not duplicate input state synchronously on `GameApp` that must be manually reset during mode switches.

2. **Forward Discrete Events Without Folding State:**
   - `GameApp::on_input` must translate discrete `GameInput` events (`KeyTransition`, `MouseTransition`, `MouseDelta`) directly into `ModeManager` updates or FIFO `ActionSender<T>` messages.
   - `UiOverlay::sync` must push discrete UI actions into the same FIFO `ActionSender<T>` channel.

3. **Zero Polling When Idle:**
   - Async scripts in `TaskSlab` must park cleanly on `ActionReceiver::next(WaitTimeout::None)` or `StateSignal::wait_until(...)` when idle.
   - Never use short sleep timeouts (such as `WaitTimeout::WorldMs(50)`) inside an action loop to poll for state changes.
   - Use `WaitTimeout::WorldMs` or `WaitTimeout::UiMs` only for real time-based mechanics (movement ticks while active, buff durations, UI auto-close timers, AI intervals).

4. **Own Transient Control State Inside Async Futures:**
   - Keep transient interaction state (such as held movement keys or AI patrol phase) private inside the `async` function state machine in `TaskSlab`.
   - Use `StateSignal::wait_until` and `futures::select_biased!` to wake active loops immediately when state changes.

## 2. Engine Architecture Principles and Crate Boundaries

See [docs/engine-architecture.md](docs/engine-architecture.md) for the complete architecture specification. All changes to `crates/engine` and `crates/game` must preserve these invariants:

- **Strict Engine vs. Game Split:** Keep `crates/engine` game-agnostic, reusable, and self-contained so an external `game` workspace can depend on it directly. Never place game-specific types, names, or assets inside `crates/engine`. Keep `crates/game` as a minimal-boilerplate sample game that demonstrates all engine features.
- **Type-Enforced Privacy:** Keep `TaskSlab`, `TaskHandle`, `SlotMap`, SoA arrays, `rapier3d` sets, and raw sync crates (`async-channel`, `event-listener`, `async-executor`) private inside `crates/engine`. Never depend on `tokio` inside `engine`.
- **`rust-gpu` Shaders Only:** Keep core GPU shaders in `crates/engine/shaders/core_shader` compiled by `crates/engine/build.rs` into Cargo `OUT_DIR`. Never write shader build artifacts into the source tree and never introduce WGSL shaders.
- **Minimal Game Boilerplate (`crates/game`):** Keep `crates/game/src/main.rs` minimal by delegating the runtime loop, windowing, rendering, and scheduling to `engine::EngineRunner::run`.
- **No Borrows Across `.await`:** Access AoS `ActorData` and `StateSignal` strictly through synchronous closures (`read_actor`, `write_actor`, `StateSignal::read`).
- **Owner-Bound Lifetimes and Four Runtime Rules:** Spawn scripts only through `attach_actor_script`, `spawn_buff`, `push_mode_with_script`, and `spawn_global_script`. Return `Option` on expired `ActorId`, drop `TaskHandle` on despawn, require `WaitTimeout` on async waits, and cancel old buffs on refresh.
- **Hybrid Dual-Executor and Coupled Physics Step:** Run `Physics -> poll TaskSlab -> flush PhysicsCommand` in one simulation step; offload heavy compute to `WorkerPool` via cloned snapshots and revalidate `ActorId` after `.await`.
- **Lossless Input and `RenderSchedule`:** Batch `winit` events losslessly on the main thread, process sequentially against `ModeManager`, expose only FIFO `ActionSender`/`ActionReceiver` (no `Notify`), and keep both `RenderSnapshot` buffers private inside `EngineRunner`.
- **No OS/Filesystem I/O in Async Tasks:** Perform checkpoint disk serialization (`SaveCodec`) only in synchronous hooks (`GameApp::on_sync`) while `ClockDomain::World` is paused.
- **Dual Clocks, Modalities, and Decoupled `UiOverlay`:** Gate `ClockDomain::World` through `ModeManager<FlatMode, StackMode>`, keep `ClockDomain::Ui` at `1.0` in all modes, preserve `StackMode` across `FlatMode` toggles, and keep `engine` independent of any UI library.

## 3. Code Style & Clippy Guidelines

### Mandatory Clippy Verification

Every workspace crate must pass Clippy without warnings:

```bash
cargo clippy --workspace --all-targets -- -D warnings
```

- **Shader Crate Exemption:** `crates/engine/shaders/core_shader` is an isolated `#![no_std]` `rust-gpu` (`spirv-std`) crate excluded from the root workspace. Host `tracing`, `std::any`, and workspace Clippy rules do not apply to it.

### Allowing `clippy::new_without_default`

Use `#[allow(clippy::new_without_default)]` locally on a struct's `new` method when implementing `Default` is semantically incorrect or `new(...)` requires parameters or non-default setup.

### Error Handling & Visibility

Root `Cargo.toml` and `clippy.toml` deny `let_underscore_must_use`, `unwrap_used`, `disallowed_methods` (`Result::expect`), and `disallowed_macros`.

Do not silently drop `Result` values with `let _ =`. Return `Result` across fallible boundaries where practical (such as `SaveCodec`, checkpoint I/O, and `EngineRunner::run`). When an error must be handled locally without propagating, log it with `tracing`.

These restrictions do not apply in tests. In `#[cfg(test)]` modules or test files, annotate with `#[allow(clippy::disallowed_methods)]` (or `#![allow(clippy::disallowed_methods)]`) to use `.expect("<message>")` and `.unwrap()` for test assertions.

**Preferred Patterns:**

1. **Standalone statements (log and continue):**

   ```rust
   if let Err(err) = result {
       tracing::error!(?err, "Operation failed");
   }
   ```

   For expected non-fatal contention (such as full bounded input channels on `try_send` or idempotent initialization like `tracing_subscriber::fmt::try_init`), log at `tracing::warn!` or `tracing::debug!` rather than `tracing::error!`.

2. **Chaining method calls (log and discard):**

   ```rust
   result
       .inspect_err(|err| tracing::error!(?err, "Operation failed"))
       .ok();
   ```

   Do not log normal async task cancellation (handled by `TaskHandle` drop), and distinguish expected `WaitTimeout` expirations (`ActionReceiver::next`, `StateSignal::wait_until`) from real wait errors (`WaitError::NoClockContext`).

3. **Theoretically impossible states:**
   Use `unreachable!` when a type signature exposes an error or `None` variant that program invariants prevent:

   ```rust
   result.unwrap_or_else(|err| unreachable!("State impossible because ...: {err:?}"));
   option.unwrap_or_else(|| unreachable!("State impossible because ..."));
   ```

4. **Fatal, unrecoverable internal invariants (`Result` types):**
   Never call `.expect("<message>")` on `Result`. Use `.unwrap_or_else` with `panic!` and include the error context (never discard with `|_|`):

   ```rust
   let value = fallible_op().unwrap_or_else(|err| panic!("Operation failed: {err:?}"));
   ```

   For non-`Debug`/non-`Display` error values (such as `Box<dyn Any>` downcasting or `Vec::try_into` array conversions), capture the error variable and log available dynamic context (for example `std::any::type_name::<T>()`, `(*err).type_id()`, or `vec.len()`).

   For `Option` values representing invariant violations, use `.expect("<message>")` for static messages or `.unwrap_or_else(|| panic!("...: {ctx}"))` when formatting dynamic context.
