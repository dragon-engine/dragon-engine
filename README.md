# Dragon Engine

3D game engine and sample RPG workspace built in Rust with `wgpu`, `winit`, `rapier3d`, and `rust-gpu` SPIR-V shaders.

## Workspace Structure

- `crates/engine`: Self-contained, game-agnostic runtime crate .
- `crates/game`: Minimal-boilerplate sample RPG demonstrating modes, reactive async scripts, UI overlay, checkpoints, and runtime mode switching.
- `crates/engine/shaders/core_shader`: Standalone `rust-gpu` (`#![no_std]`) shader crate compiled by `engine` build scripts into `OUT_DIR`.

## Documentation

- [Engine Architecture Principles](docs/engine-architecture.md): Core safety invariants, memory layout, dual executors, clocks, input pipeline, and UI abstraction.
- [Graphics & Streaming Architecture](docs/graphics-architecture.md): Finalized 12-pass compute-centric clustered forward, unified hardware ray-query, and zero-loading-screen streaming specifications.
- [Graphics Implementation Plan](docs/graphics-implementation-plan.md): Phased execution roadmap, codebase readiness audit, verification gates, and progress tracker.
- [Project Agent Rules](AGENTS.md): Mandatory reactive architecture and crate boundary rules.

## Build and Test

Run standard Cargo commands from the repository root (`cargo test --workspace`, `cargo run -p game`).
