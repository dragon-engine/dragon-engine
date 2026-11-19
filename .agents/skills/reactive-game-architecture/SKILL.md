---
name: reactive-game-architecture
description: Enforces reactive, event-driven gameplay architecture and pure domain state in Dragon Engine (`crates/game` and `crates/engine`). Use when implementing gameplay systems, input handling, async scripts, UI overlays, or reviewing state models.
---

# Reactive Game Architecture and Pure State

Apply this skill whenever you add or modify gameplay logic, input pipelines, async actor scripts, or UI interactions in `crates/game` and `crates/engine`.

## Core Rule

State must contain only domain data. Control flow, transient input bookkeeping, and time-based loops belong inside async `TaskSlab` futures that react to `ActionReceiver` messages and `StateSignal` changes.

## Design Checklist

1. **Audit `GameApp` and `ActorData` structs:**
   - Keep only domain fields (`ActorId`, HP, potions, quest stage, buff remaining time, camera orientation).
   - Remove any input-tracking booleans (`move_w`, `is_firing`, `key_down`) or manual reset logic on mode transitions.

2. **Route discrete inputs in `GameApp::on_input`:**
   - Translate `GameInput::Key` (`KeyTransition` press and release) and `GameInput::MouseButton` directly into `ModeManager` transitions or `ActionSender::try_send` messages.
   - Forward key releases unconditionally or send a discrete clear action on mode exit so held inputs do not stick.
   - Route `UiOverlay::sync` widget messages into the same FIFO `ActionSender` channel.

3. **Separate event dispatch from active time-based loops:**
   - Park event dispatchers on `rx.next(WaitTimeout::None).await` with zero idle timeouts.
   - Store transient interaction state (for example `HeldMoveKeys`) as local variables inside the `async` future.
   - Publish derived domain intent (for example `StateSignal<glam::Vec3>` movement direction) so downstream coroutines react immediately.

4. **Use `StateSignal::wait_until` and `select_biased!` for continuous mechanics:**
   - Park active loops on `signal.wait_until(WaitTimeout::None, ...)` whenever inactive.
   - While active, race a domain clock step (`ctx.sleep(WaitTimeout::WorldMs(16))`) against `signal.wait_until(...)` via `engine::select_biased!`.

5. **Enforce engine safety and reusability boundaries:**
   - Keep `crates/engine` strictly game-agnostic and self-contained (runtime, core `rust-gpu` shaders, `winit`/Vulkan 1.3 `ash` runner, physics, async primitives) so external `game` workspaces can reuse it without boilerplate.
   - Keep `crates/game` as the feature-complete sample game without leaking game concepts into `crates/engine`.
   - Never hold AoS or `StateSignal` borrows across `.await`; use synchronous closures (`read_actor`, `write_actor`, `StateSignal::read`).
   - Revalidate `ActorId` handles (`Option`) after `.await` or background `spawn_compute` calls.
   - Keep OS and filesystem I/O (such as `SaveCodec` checkpoints) inside synchronous `GameApp::on_sync` while `ClockDomain::World` is paused.

## Reference Pattern

```rust
// Input handler: forward discrete events only, hold zero input flags on GameApp.
match input {
    GameInput::Key(k) if !k.repeat => {
        let pressed = k.state == ElementState::Pressed;
        let _ = ctx.action_sender().try_send(RpgAction::MoveKey { key, pressed });
    }
    _ => {}
}

// Event dispatcher script: parks with zero polling when idle.
while let Ok(action) = rx.next(WaitTimeout::None).await {
    match action {
        RpgAction::MoveKey { key, pressed } => {
            held_keys.set(key, pressed);
            move_dir_signal.set(held_keys.direction());
        }
        RpgAction::ClearMovement => {
            held_keys.clear();
            move_dir_signal.set(glam::Vec3::ZERO);
        }
        // ...
    }
}

// Active movement coroutine: zero wakeups when stationary, immediate wakeup on direction change.
loop {
    let Ok(mut dir) = move_dir_signal
        .wait_until(WaitTimeout::None, |d| (*d != glam::Vec3::ZERO).then_some(*d))
        .await
    else {
        return;
    };

    while dir != glam::Vec3::ZERO {
        if ctx.is_world_active() {
            ctx.push_physics_command(PhysicsCommand::ApplyImpulse {
                actor: player_id,
                impulse: dir * 1.4,
            });
        }
        let current_dir = dir;
        let mut step = std::pin::pin!(ctx.sleep(WaitTimeout::WorldMs(16)).fuse());
        let mut changed = std::pin::pin!(
            move_dir_signal
                .wait_until(WaitTimeout::None, move |d| (*d != current_dir).then_some(*d))
                .fuse()
        );
        select_biased! {
            res = changed => dir = match res { Ok(next) => next, Err(_) => return },
            _ = step => {}
        }
    }
}
```
