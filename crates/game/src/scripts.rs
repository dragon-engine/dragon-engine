use std::sync::atomic::Ordering;

use engine::{
    ActorId, EffectId, PhysicsCommand, ScriptContext, StateSignal, WaitTimeout, select_biased,
};
use futures::FutureExt;
use fx_fresnel_rim::FresnelRimParams;

use crate::app::{
    BUFF_REGEN_KEY, FlatMode, MoveKey, QuestStage, RpgAction, RpgGame, SharedRpgSignals, StackMode,
};

#[derive(Default)]
struct HeldMoveKeys {
    forward: bool,
    backward: bool,
    left: bool,
    right: bool,
}

impl HeldMoveKeys {
    fn set(&mut self, key: MoveKey, pressed: bool) {
        match key {
            MoveKey::Forward => self.forward = pressed,
            MoveKey::Backward => self.backward = pressed,
            MoveKey::Left => self.left = pressed,
            MoveKey::Right => self.right = pressed,
        }
    }

    fn clear(&mut self) {
        *self = Self::default();
    }

    fn direction(&self) -> glam::Vec3 {
        let mut dir = glam::Vec3::ZERO;
        if self.forward {
            dir.z -= 1.0;
        }
        if self.backward {
            dir.z += 1.0;
        }
        if self.left {
            dir.x -= 1.0;
        }
        if self.right {
            dir.x += 1.0;
        }
        dir.normalize_or_zero()
    }
}

pub(crate) async fn player_movement_script(
    ctx: ScriptContext<RpgGame>,
    player_id: ActorId,
    move_dir_signal: StateSignal<glam::Vec3>,
    facing_yaw_signal: StateSignal<f32>,
) {
    loop {
        // Park without polling while no movement keys are held.
        let Ok(mut local_dir) = move_dir_signal
            .wait_until(
                WaitTimeout::None,
                |d| (*d != glam::Vec3::ZERO).then_some(*d),
            )
            .await
        else {
            return;
        };

        while local_dir != glam::Vec3::ZERO {
            if ctx.is_world_active() {
                let yaw = facing_yaw_signal.read(|y| *y);
                let world_dir = crate::app::local_move_to_world(
                    local_dir, yaw,
                );
                ctx.push_physics_command(
                    PhysicsCommand::ApplyImpulse {
                        actor: player_id,
                        impulse: world_dir * 1.4,
                    },
                );
            }

            let current_dir = local_dir;
            let mut step = std::pin::pin!(ctx.sleep(WaitTimeout::WorldMs(16)).fuse());
            let mut dir_changed = std::pin::pin!(
                move_dir_signal
                    .wait_until(
                        WaitTimeout::None,
                        move |d| (*d != current_dir).then_some(*d),
                    )
                    .fuse()
            );

            select_biased! {
                res = dir_changed => {
                    match res {
                        Ok(next) => local_dir = next,
                        Err(_) => return,
                    }
                }
                _ = step => {}
            }

            if ctx
                .read_actor(
                    player_id,
                    |a| a.hp,
                )
                .is_none()
            {
                return;
            }
        }
    }
}

pub(crate) async fn player_script(
    ctx: ScriptContext<RpgGame>,
    player_id: ActorId,
    goblin_id: Option<ActorId>,
    tree_id: Option<ActorId>,
    rim_effect: EffectId,
    signals: SharedRpgSignals,
) {
    let mut held_keys = HeldMoveKeys::default();
    let rx = ctx.receiver().clone();

    while let Ok(action) = rx.next(WaitTimeout::None).await {
        match action {
            RpgAction::MoveKey { key, pressed } => {
                held_keys.set(
                    key, pressed,
                );
                signals.move_dir.set(held_keys.direction());
            }
            RpgAction::ClearMovement => {
                held_keys.clear();
                signals.move_dir.set(glam::Vec3::ZERO);
            }
            RpgAction::SetFacingYaw(yaw) => {
                ctx.write_actor(
                    player_id,
                    |p| {
                        p.facing_yaw = yaw;
                    },
                );
                ctx.push_physics_command(
                    PhysicsCommand::SetRotation {
                        actor: player_id,
                        rotation: crate::app::facing_rotation(yaw),
                    },
                );
            }
            RpgAction::Attack => {
                if ctx.is_world_active() {
                    perform_player_attack(
                        &ctx, player_id, &signals,
                    )
                    .await;
                }
            }
            RpgAction::UsePotion => {
                // Frozen-world items: AoS heal applies immediately even when World clock is 0.0,
                // while the regen buff ticks on ClockDomain::World once play resumes.
                let used = ctx.write_actor(
                    player_id,
                    |p| {
                        if p.potions > 0 && p.hp > 0 {
                            p.potions -= 1;
                            p.hp = (p.hp + 30).min(p.max_hp);
                            p.regen_remaining_ms = 5000;
                            true
                        } else {
                            false
                        }
                    },
                );
                if used == Some(true) {
                    signals.status_banner.set(
                        String::from(
                            "Potion used: +30 HP now, +5s Regen buff queued on World clock",
                        ),
                    );
                    // Re-spawning with BUFF_REGEN_KEY cancels any old buff TaskHandle automatically.
                    ctx.spawn_buff(
                        player_id,
                        BUFF_REGEN_KEY,
                        move |bctx| {
                            regen_buff_script(
                                bctx, player_id, 5000, rim_effect,
                            )
                        },
                    );
                }
            }
            RpgAction::ToggleInventory => {
                if ctx.flat_mode() == Some(FlatMode::InGame) {
                    held_keys.clear();
                    signals.move_dir.set(glam::Vec3::ZERO);
                    match ctx.stack_top() {
                        Some(StackMode::Play) => ctx.push_mode(StackMode::Inventory),
                        Some(StackMode::Inventory) => {
                            ctx.pop_mode();
                        }
                        Some(StackMode::ConfirmDiscard) => {
                            ctx.pop_mode();
                        }
                        None => {}
                    }
                }
            }
            RpgAction::DiscardPotionPrompt => {
                if ctx.stack_top() == Some(StackMode::Inventory) {
                    let sig = signals.clone();
                    ctx.push_mode_with_script(
                        StackMode::ConfirmDiscard,
                        move |mctx| async move {
                            // Mode-scoped script: auto-cancels confirmation modal after 6s UI time,
                            // or cancels immediately if the player closes the mode earlier.
                            if mctx.sleep(WaitTimeout::UiMs(6000)).await.is_ok()
                                && mctx.stack_top() == Some(StackMode::ConfirmDiscard)
                            {
                                mctx.pop_mode();
                                sig.status_banner
                                    .set(String::from("Discard prompt timed out on Ui clock"));
                            }
                        },
                    );
                }
            }
            RpgAction::ConfirmDiscardPotion => {
                ctx.write_actor(
                    player_id,
                    |p| {
                        p.potions = p.potions.saturating_sub(1);
                    },
                );
                if ctx.stack_top() == Some(StackMode::ConfirmDiscard) {
                    ctx.pop_mode();
                }
                signals
                    .status_banner
                    .set(String::from("Discarded 1 potion from inventory"));
            }
            RpgAction::CancelModal => {
                if ctx.pop_mode().is_none() && ctx.flat_mode() == Some(FlatMode::MainMenu) {
                    ctx.set_flat_mode(FlatMode::InGame);
                }
            }
            RpgAction::ToggleMainMenu => {
                held_keys.clear();
                signals.move_dir.set(glam::Vec3::ZERO);
                let next = match ctx.flat_mode() {
                    Some(FlatMode::MainMenu) => FlatMode::InGame,
                    _ => FlatMode::MainMenu,
                };
                ctx.set_flat_mode(next);
            }
            RpgAction::SetSchedule(schedule) => {
                ctx.set_render_schedule(schedule);
                signals
                    .status_banner
                    .set(format!("RenderSchedule switched to {schedule:?}"));
            }
            RpgAction::SaveCheckpoint => {
                signals.save_requested.store(
                    true,
                    Ordering::Release,
                );
            }
            RpgAction::LoadCheckpoint => {
                signals.load_requested.store(
                    true,
                    Ordering::Release,
                );
            }
            RpgAction::Quit => {
                ctx.request_exit();
            }
        }

        // Keep references used for compiler clarity when checkpoint restores inspect targets.
        let _ = (
            goblin_id, tree_id,
        );
    }
}

const ATTACK_CONE_HALF_DEG: f32 = 15.0;

pub(crate) fn check_attack_cone(
    origin: glam::Vec3,
    target_pos: glam::Vec3,
    forward: glam::Vec3,
) -> Result<(), f32> {
    let mut to_target = target_pos - origin;
    to_target.y = 0.0;
    let dist_sq = to_target.length_squared();
    if dist_sq <= 1e-6 {
        return Ok(());
    }
    let dir_to_target = to_target / dist_sq.sqrt();
    let cos_angle = forward.dot(dir_to_target).clamp(
        -1.0, 1.0,
    );
    let min_cos = ATTACK_CONE_HALF_DEG.to_radians().cos();
    if cos_angle >= min_cos {
        Ok(())
    } else {
        Err(cos_angle.acos().to_degrees())
    }
}

async fn perform_player_attack(
    ctx: &ScriptContext<RpgGame>,
    player_id: ActorId,
    signals: &SharedRpgSignals,
) {
    let Some(origin) = ctx.spatial(|sq| sq.actor_position(player_id)).flatten() else {
        return;
    };
    let attack_power = ctx
        .read_actor(
            player_id,
            |p| p.attack_power,
        )
        .unwrap_or(10);
    let yaw = signals.facing_yaw.read(|y| *y);
    let forward = crate::app::facing_forward(yaw).normalize();

    let nearby = ctx
        .spatial(
            |sq| {
                sq.overlap_sphere(
                    origin,
                    3.2,
                    Some(player_id),
                )
            },
        )
        .unwrap_or_default();

    let mut closest_miss: Option<(
        String,
        f32,
    )> = None;

    for target_id in nearby {
        let Some(target_name) = ctx
            .read_actor(
                target_id,
                |target| {
                    matches!(
                        target.role,
                        crate::app::ActorRole::Goblin | crate::app::ActorRole::CorruptedTree
                    )
                    .then(|| target.name.clone())
                },
            )
            .flatten()
        else {
            continue;
        };

        let Some(target_pos) = ctx.spatial(|sq| sq.actor_position(target_id)).flatten() else {
            continue;
        };
        if let Err(angle_deg) = check_attack_cone(
            origin, target_pos, forward,
        ) {
            if closest_miss
                .as_ref()
                .is_none_or(|(_, best_deg)| angle_deg < *best_deg)
            {
                closest_miss = Some(
                    (
                        target_name,
                        angle_deg,
                    ),
                );
            }
            continue;
        }

        let role_and_hp = ctx.write_actor(
            target_id,
            |target| {
                if matches!(
                    target.role,
                    crate::app::ActorRole::Goblin | crate::app::ActorRole::CorruptedTree
                ) {
                    target.hp = (target.hp - attack_power).max(0);
                    Some(
                        (
                            target.name.clone(),
                            target.hp,
                        ),
                    )
                } else {
                    None
                }
            },
        );

        if let Some(Some((name, new_hp))) = role_and_hp {
            ctx.push_physics_command(
                PhysicsCommand::SetMeshColor {
                    actor: target_id,
                    color: [1.0, 0.85, 0.25],
                },
            );
            signals
                .status_banner
                .set(format!("Hit {name} for {attack_power} dmg (HP {new_hp})"));

            if new_hp <= 0 {
                // Despawning cancels all attached Actor tasks cleanly.
                ctx.despawn_actor(target_id);
                signals
                    .status_banner
                    .set(format!("{name} defeated and despawned!"));
            } else {
                let restore_color = if name.contains("Goblin") {
                    [0.85, 0.25, 0.25]
                } else {
                    [0.46, 0.30, 0.18]
                };
                if let Err(err) = ctx.sleep(WaitTimeout::WorldMs(90)).await {
                    tracing::error!(
                        ?err,
                        "Attack flash sleep failed"
                    );
                }
                ctx.push_physics_command(
                    PhysicsCommand::SetMeshColor {
                        actor: target_id,
                        color: restore_color,
                    },
                );
            }
            return;
        }
    }

    if let Some((name, angle_deg)) = closest_miss {
        signals.status_banner.set(
            format!("Missed {name} ({angle_deg:.0} deg off front; effective within +-15 deg)"),
        );
    } else {
        signals
            .status_banner
            .set(String::from("Attack missed: no target in front (+-15 deg)"));
    }
}

pub(crate) async fn regen_buff_script(
    ctx: ScriptContext<RpgGame>,
    player_id: ActorId,
    mut remaining_ms: u64,
    rim_effect: EffectId,
) {
    const TICK_MS: u64 = 1000;
    ctx.push_physics_command(
        PhysicsCommand::SetMeshEffect {
            actor: player_id,
            effect: Some(
                FresnelRimParams::new(
                    [0.20, 0.95, 0.65],
                    0.95,
                    1.8,
                )
                .with_pulse(5.5)
                .attach(rim_effect),
            ),
        },
    );
    while remaining_ms > 0 {
        let step = remaining_ms.min(TICK_MS);
        if ctx.sleep(WaitTimeout::WorldMs(step)).await.is_err() {
            return;
        }
        remaining_ms = remaining_ms.saturating_sub(step);
        // Revalidate ActorId after .await via Option return.
        let alive = ctx.write_actor(
            player_id,
            |p| {
                p.hp = (p.hp + 5).min(p.max_hp);
                p.regen_remaining_ms = remaining_ms;
            },
        );
        if alive.is_none() {
            return;
        }
    }
    ctx.push_physics_command(
        PhysicsCommand::SetMeshEffect {
            actor: player_id,
            effect: None,
        },
    );
}

#[derive(Clone)]
struct PathSnapshot {
    self_pos: glam::Vec3,
    target_pos: glam::Vec3,
}

pub(crate) async fn goblin_ai_script(
    ctx: ScriptContext<RpgGame>,
    goblin_id: ActorId,
    player_id: ActorId,
    signals: SharedRpgSignals,
) {
    // Private AI variables packed directly into the TaskSlab Future state machine.
    let mut patrol_phase: f32 = 0.0;

    loop {
        if ctx.sleep(WaitTimeout::WorldMs(250)).await.is_err() {
            return;
        }

        let positions = ctx.spatial(
            |sq| {
                let g_pos = sq.actor_position(goblin_id)?;
                let p_pos = sq.actor_position(player_id)?;
                Some(
                    PathSnapshot {
                        self_pos: g_pos,
                        target_pos: p_pos,
                    },
                )
            },
        );

        let Some(Some(snapshot)) = positions else {
            return;
        };

        // Heavy compute steering/pathfinding dispatched to the background async-executor pool
        // with a cloned snapshot; ActorId is revalidated after .await returns.
        let Some(task) = ctx.spawn_compute(
            snapshot.clone(),
            |snap| async move {
                let delta = snap.target_pos - snap.self_pos;
                let flat = glam::Vec3::new(
                    delta.x, 0.0, delta.z,
                );
                let dist = flat.length();
                (
                    flat.normalize_or_zero(),
                    dist,
                )
            },
        ) else {
            return;
        };

        let (dir, dist) = task.await;

        // Revalidate both ActorIds after .await.
        if ctx
            .read_actor(
                goblin_id,
                |g| g.hp,
            )
            .unwrap_or(0)
            <= 0
        {
            return;
        }
        if ctx
            .read_actor(
                player_id,
                |p| p.hp,
            )
            .is_none()
        {
            return;
        }

        if dist < 2.2 {
            let dmg = ctx
                .read_actor(
                    goblin_id,
                    |g| g.attack_power,
                )
                .unwrap_or(6);
            if let Some(hp_left) = ctx.write_actor(
                player_id,
                |p| {
                    p.hp = (p.hp - dmg).max(0);
                    p.hp
                },
            ) {
                signals
                    .status_banner
                    .set(format!("Goblin struck Knight for {dmg} dmg (HP {hp_left})"));
            }
            if let Err(err) = ctx.sleep(WaitTimeout::WorldMs(750)).await {
                tracing::error!(
                    ?err,
                    "Goblin attack cooldown sleep failed"
                );
            }
        } else if dist < 13.0 {
            ctx.push_physics_command(
                PhysicsCommand::ApplyImpulse {
                    actor: goblin_id,
                    impulse: dir * 4.5,
                },
            );
        } else {
            patrol_phase += 0.6;
            let orbit = glam::Vec3::new(
                patrol_phase.cos(),
                0.0,
                patrol_phase.sin(),
            );
            ctx.push_physics_command(
                PhysicsCommand::ApplyImpulse {
                    actor: goblin_id,
                    impulse: orbit * 2.0,
                },
            );
        }
    }
}

pub(crate) async fn quest_stage_script(
    ctx: ScriptContext<RpgGame>,
    goblin_id: Option<ActorId>,
    tree_id: Option<ActorId>,
    mut stage: QuestStage,
    rim_effect: EffectId,
    signals: SharedRpgSignals,
) {
    loop {
        match stage {
            QuestStage::HuntGoblin => {
                // Poll or await StateSignal/WorldMs cooperatively with select_biased!.
                let goblin_alive = goblin_id
                    .and_then(
                        |id| {
                            ctx.read_actor(
                                id,
                                |g| g.hp > 0,
                            )
                        },
                    )
                    .unwrap_or(false);
                if !goblin_alive {
                    stage = QuestStage::PurifyTree;
                    signals.quest_stage.set(stage);
                    if let Some(tid) = tree_id {
                        ctx.push_physics_command(
                            PhysicsCommand::SetMeshEffect {
                                actor: tid,
                                effect: Some(
                                    FresnelRimParams::new(
                                        [0.85, 0.35, 1.0],
                                        0.95,
                                        1.8,
                                    )
                                    .with_pulse(4.0)
                                    .attach(rim_effect),
                                ),
                            },
                        );
                    }
                    signals
                        .status_banner
                        .set(String::from("Quest advanced: Strike the Corrupted Tree!"));
                } else {
                    let mut timer = std::pin::pin!(ctx.sleep(WaitTimeout::WorldMs(250)).fuse());
                    let mut wait_sig = std::pin::pin!(
                        signals
                            .quest_stage
                            .wait_until(
                                WaitTimeout::WorldMs(500),
                                |s| { (*s != QuestStage::HuntGoblin).then_some(*s) }
                            )
                            .fuse()
                    );
                    select_biased! {
                        res = wait_sig => {
                            if let Ok(next_stage) = res {
                                stage = next_stage;
                            }
                        }
                        _ = timer => {}
                    }
                }
            }
            QuestStage::PurifyTree => {
                let tree_alive = tree_id
                    .and_then(
                        |id| {
                            ctx.read_actor(
                                id,
                                |t| t.hp > 0,
                            )
                        },
                    )
                    .unwrap_or(false);
                if !tree_alive {
                    stage = QuestStage::Victory;
                    signals.quest_stage.set(stage);
                    signals
                        .status_banner
                        .set(String::from("Victory! Meadow purified."));
                    return;
                }
                if ctx.sleep(WaitTimeout::WorldMs(250)).await.is_err() {
                    return;
                }
            }
            QuestStage::Victory => return,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn attack_cone_accepts_within_15_deg_and_rejects_outside() {
        let origin = glam::Vec3::ZERO;
        let forward = glam::Vec3::NEG_Z;

        // 10 degrees right of -Z -> hit.
        let ten_deg = 10.0_f32.to_radians();
        let target_hit = glam::Vec3::new(
            ten_deg.sin() * 2.0,
            0.0,
            -ten_deg.cos() * 2.0,
        );
        assert!(
            check_attack_cone(
                origin, target_hit, forward,
            )
            .is_ok()
        );

        // 25 degrees left of -Z -> miss with ~25 deg angle reported.
        let twenty_five_deg = -25.0_f32.to_radians();
        let target_miss = glam::Vec3::new(
            twenty_five_deg.sin() * 2.0,
            0.0,
            -twenty_five_deg.cos() * 2.0,
        );
        assert!(
            matches!(
                check_attack_cone(
                    origin,
                    target_miss,
                    forward,
                ),
                Err(deg) if (deg - 25.0).abs() < 0.1
            )
        );
    }
}
