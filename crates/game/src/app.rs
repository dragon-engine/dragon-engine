use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use engine::{
    ActionSender, ActorId, ActorSpawnDesc, BodyKind, CameraSnapshot, EffectId, EngineContext,
    GameApp, GameInput, GizmoBox, InputContext, RenderSchedule, StateSignal, SyncContext,
    TimerCheckpoint, WorldCheckpoint, WorldReadView, read_checkpoint_file, write_checkpoint_file,
};
use fx_fresnel_rim::FresnelRimParams;
use serde::{Deserialize, Serialize};
use winit::event::{ElementState, MouseButton};
use winit::keyboard::KeyCode;

use crate::save_codec::PostcardSaveCodec;
use crate::scripts;

pub(crate) const BUFF_REGEN_KEY: u64 = 1;
const CHECKPOINT_PATH: &str = "saves/rpg_checkpoint.bin";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) enum FlatMode {
    MainMenu,
    InGame,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) enum StackMode {
    Play,
    Inventory,
    ConfirmDiscard,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) enum ActorRole {
    Player,
    Goblin,
    CorruptedTree,
    Obstacle,
    Ground,
}

pub(crate) fn facing_forward(yaw: f32) -> glam::Vec3 {
    glam::Vec3::new(
        yaw.sin(),
        0.0,
        -yaw.cos(),
    )
}

pub(crate) fn facing_right(yaw: f32) -> glam::Vec3 {
    glam::Vec3::new(
        yaw.cos(),
        0.0,
        yaw.sin(),
    )
}

pub(crate) fn facing_rotation(yaw: f32) -> glam::Quat {
    glam::Quat::from_rotation_y(-yaw)
}

pub(crate) fn local_move_to_world(local_dir: glam::Vec3, yaw: f32) -> glam::Vec3 {
    let forward = facing_forward(yaw);
    let right = facing_right(yaw);
    // HeldMoveKeys encodes W as -Z, S as +Z, A as -X, D as +X.
    (right * local_dir.x + forward * (-local_dir.z)).normalize_or_zero()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct RpgActorData {
    pub name: String,
    pub role: ActorRole,
    pub hp: i32,
    pub max_hp: i32,
    pub potions: u32,
    pub attack_power: i32,
    pub regen_remaining_ms: u64,
    #[serde(default)]
    pub facing_yaw: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) enum QuestStage {
    HuntGoblin,
    PurifyTree,
    Victory,
}

impl QuestStage {
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::HuntGoblin => "Stage 1: Defeat the Goblin Scout",
            Self::PurifyTree => "Stage 2: Strike the Corrupted Tree",
            Self::Victory => "Stage 3: Meadow Purified (Victory)",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MoveKey {
    Forward,
    Backward,
    Left,
    Right,
}

#[derive(Debug, Clone)]
pub(crate) enum RpgAction {
    MoveKey { key: MoveKey, pressed: bool },
    ClearMovement,
    SetFacingYaw(f32),
    Attack,
    UsePotion,
    DiscardPotionPrompt,
    ConfirmDiscardPotion,
    CancelModal,
    ToggleInventory,
    ToggleMainMenu,
    SetSchedule(RenderSchedule),
    SaveCheckpoint,
    LoadCheckpoint,
    Quit,
}

#[derive(Debug, Clone)]
pub(crate) struct SharedRpgSignals {
    pub quest_stage: StateSignal<QuestStage>,
    pub status_banner: StateSignal<String>,
    pub move_dir: StateSignal<glam::Vec3>,
    pub facing_yaw: StateSignal<f32>,
    pub save_requested: Arc<AtomicBool>,
    pub load_requested: Arc<AtomicBool>,
}

impl Default for SharedRpgSignals {
    fn default() -> Self {
        Self {
            quest_stage: StateSignal::new(QuestStage::HuntGoblin),
            status_banner: StateSignal::new(
                String::from("WASD move | Mouse turn | Space/Click attack (+-15 deg) | Esc menu"),
            ),
            move_dir: StateSignal::new(glam::Vec3::ZERO),
            facing_yaw: StateSignal::new(0.0),
            save_requested: Arc::new(AtomicBool::new(false)),
            load_requested: Arc::new(AtomicBool::new(false)),
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct RpgSnapshot {
    pub flat_mode: FlatMode,
    pub stack_modes: Vec<StackMode>,
    pub player_hp: i32,
    pub player_max_hp: i32,
    pub player_potions: u32,
    pub player_regen_ms: u64,
    pub goblin_hp: Option<i32>,
    pub tree_hp: Option<i32>,
    pub quest_stage: QuestStage,
    pub status_banner: String,
}

impl Default for RpgSnapshot {
    fn default() -> Self {
        Self {
            flat_mode: FlatMode::InGame,
            stack_modes: vec![StackMode::Play],
            player_hp: 100,
            player_max_hp: 100,
            player_potions: 3,
            player_regen_ms: 0,
            goblin_hp: Some(45),
            tree_hp: Some(80),
            quest_stage: QuestStage::HuntGoblin,
            status_banner: String::new(),
        }
    }
}

pub(crate) struct RpgGame {
    pub player_id: Option<ActorId>,
    pub goblin_id: Option<ActorId>,
    pub tree_id: Option<ActorId>,
    pub signals: SharedRpgSignals,
    rim_effect: EffectId,
    facing_yaw: f32,
}

impl RpgGame {
    pub(crate) fn new(rim_effect: EffectId) -> Self {
        Self {
            player_id: None,
            goblin_id: None,
            tree_id: None,
            signals: SharedRpgSignals::default(),
            rim_effect,
            facing_yaw: 0.0,
        }
    }

    fn world_active(flat: &FlatMode, stack: &[StackMode]) -> bool {
        *flat == FlatMode::InGame && stack.last() == Some(&StackMode::Play)
    }
}

fn send_action(sender: &ActionSender<RpgAction>, action: RpgAction) {
    if let Err(err) = sender.try_send(action) {
        tracing::warn!(
            ?err,
            "Failed to enqueue input action"
        );
    }
}

impl GameApp for RpgGame {
    type Action = RpgAction;
    type ActorData = RpgActorData;
    type FlatMode = FlatMode;
    type SnapshotData = RpgSnapshot;
    type StackMode = StackMode;

    fn initial_modes() -> (
        Self::FlatMode,
        Self::StackMode,
        fn(&Self::FlatMode, &[Self::StackMode]) -> bool,
    ) {
        (
            FlatMode::InGame,
            StackMode::Play,
            Self::world_active,
        )
    }

    fn init(&mut self, ctx: &mut EngineContext<Self>) {
        // Ground slab.
        ctx.spawn_actor(
            ActorSpawnDesc {
                position: glam::Vec3::new(
                    0.0, -0.5, 0.0,
                ),
                rotation: glam::Quat::IDENTITY,
                half_extents: glam::Vec3::new(
                    18.0, 0.5, 18.0,
                ),
                color: [0.16, 0.28, 0.18],
                effect: None,
                body_kind: BodyKind::Fixed,
                mass: 100.0,
                data: RpgActorData {
                    name: String::from("Meadow Floor"),
                    role: ActorRole::Ground,
                    hp: 9999,
                    max_hp: 9999,
                    potions: 0,
                    attack_power: 0,
                    regen_remaining_ms: 0,
                    facing_yaw: 0.0,
                },
            },
        );

        // Stone pillars for physics collision and pathfinding obstacles.
        let pillar_positions = [
            glam::Vec3::new(
                -3.5, 1.0, -2.5,
            ),
            glam::Vec3::new(
                3.5, 1.0, -1.5,
            ),
            glam::Vec3::new(
                0.0, 1.0, -5.0,
            ),
        ];
        for (i, pos) in pillar_positions.into_iter().enumerate() {
            ctx.spawn_actor(
                ActorSpawnDesc {
                    position: pos,
                    rotation: glam::Quat::IDENTITY,
                    half_extents: glam::Vec3::new(
                        0.9, 1.2, 0.9,
                    ),
                    color: [0.42, 0.45, 0.50],
                    effect: None,
                    body_kind: BodyKind::Fixed,
                    mass: 50.0,
                    data: RpgActorData {
                        name: format!(
                            "Stone Pillar {}",
                            i + 1
                        ),
                        role: ActorRole::Obstacle,
                        hp: 9999,
                        max_hp: 9999,
                        potions: 0,
                        attack_power: 0,
                        regen_remaining_ms: 0,
                        facing_yaw: 0.0,
                    },
                },
            );
        }

        let player_id = ctx.spawn_actor(
            ActorSpawnDesc {
                position: glam::Vec3::new(
                    0.0, 1.0, 5.0,
                ),
                rotation: glam::Quat::IDENTITY,
                half_extents: glam::Vec3::new(
                    0.6, 0.9, 0.6,
                ),
                color: [0.25, 0.62, 0.95],
                effect: None,
                body_kind: BodyKind::Dynamic,
                mass: 2.0,
                data: RpgActorData {
                    name: String::from("Knight"),
                    role: ActorRole::Player,
                    hp: 100,
                    max_hp: 100,
                    potions: 3,
                    attack_power: 18,
                    regen_remaining_ms: 0,
                    facing_yaw: 0.0,
                },
            },
        );
        self.player_id = Some(player_id);

        let goblin_id = ctx.spawn_actor(
            ActorSpawnDesc {
                position: glam::Vec3::new(
                    -4.5, 1.0, -6.5,
                ),
                rotation: glam::Quat::IDENTITY,
                half_extents: glam::Vec3::new(
                    0.55, 0.85, 0.55,
                ),
                color: [0.85, 0.25, 0.25],
                effect: Some(
                    FresnelRimParams::new(
                        [1.0, 0.30, 0.20],
                        0.95,
                        1.8,
                    )
                    .with_pulse(4.0)
                    .attach(self.rim_effect),
                ),
                body_kind: BodyKind::Dynamic,
                mass: 1.8,
                data: RpgActorData {
                    name: String::from("Goblin Scout"),
                    role: ActorRole::Goblin,
                    hp: 45,
                    max_hp: 45,
                    potions: 0,
                    attack_power: 8,
                    regen_remaining_ms: 0,
                    facing_yaw: 0.0,
                },
            },
        );
        self.goblin_id = Some(goblin_id);

        let tree_id = ctx.spawn_actor(
            ActorSpawnDesc {
                position: glam::Vec3::new(
                    5.5, 1.8, -7.0,
                ),
                rotation: glam::Quat::IDENTITY,
                half_extents: glam::Vec3::new(
                    1.0, 1.8, 1.0,
                ),
                color: [0.46, 0.30, 0.18],
                effect: None,
                body_kind: BodyKind::Fixed,
                mass: 80.0,
                data: RpgActorData {
                    name: String::from("Corrupted Tree"),
                    role: ActorRole::CorruptedTree,
                    hp: 80,
                    max_hp: 80,
                    potions: 0,
                    attack_power: 0,
                    regen_remaining_ms: 0,
                    facing_yaw: 0.0,
                },
            },
        );
        self.tree_id = Some(tree_id);

        let signals = self.signals.clone();
        let rim_effect = self.rim_effect;
        ctx.attach_actor_script(
            player_id,
            move |sc| {
                scripts::player_script(
                    sc,
                    player_id,
                    Some(goblin_id),
                    Some(tree_id),
                    rim_effect,
                    signals,
                )
            },
        );
        let move_signal = self.signals.move_dir.clone();
        let yaw_signal = self.signals.facing_yaw.clone();
        ctx.attach_actor_script(
            player_id,
            move |sc| {
                scripts::player_movement_script(
                    sc,
                    player_id,
                    move_signal,
                    yaw_signal,
                )
            },
        );

        let goblin_signals = self.signals.clone();
        ctx.attach_actor_script(
            goblin_id,
            move |sc| {
                scripts::goblin_ai_script(
                    sc,
                    goblin_id,
                    player_id,
                    goblin_signals,
                )
            },
        );

        let quest_signals = self.signals.clone();
        let quest_rim = self.rim_effect;
        ctx.spawn_global_script(
            move |sc| {
                scripts::quest_stage_script(
                    sc,
                    Some(goblin_id),
                    Some(tree_id),
                    QuestStage::HuntGoblin,
                    quest_rim,
                    quest_signals,
                )
            },
        );
    }

    fn world_scale(&self) -> f32 {
        1.0
    }

    fn on_input(&mut self, input: GameInput, ctx: &mut InputContext<Self>) {
        match input {
            GameInput::Key(k) if !k.repeat => {
                let pressed = k.state == ElementState::Pressed;
                match k.key {
                    KeyCode::Escape if pressed => {
                        let next = match ctx.modes().flat() {
                            FlatMode::MainMenu => FlatMode::InGame,
                            FlatMode::InGame => FlatMode::MainMenu,
                        };
                        ctx.set_flat_mode(next);
                        send_action(
                            ctx.action_sender(),
                            RpgAction::ClearMovement,
                        );
                    }
                    KeyCode::KeyI if pressed && *ctx.modes().flat() == FlatMode::InGame => {
                        send_action(
                            ctx.action_sender(),
                            RpgAction::ToggleInventory,
                        );
                        send_action(
                            ctx.action_sender(),
                            RpgAction::ClearMovement,
                        );
                    }
                    KeyCode::KeyH if pressed => {
                        send_action(
                            ctx.action_sender(),
                            RpgAction::UsePotion,
                        );
                    }
                    KeyCode::F5 if pressed => {
                        send_action(
                            ctx.action_sender(),
                            RpgAction::SaveCheckpoint,
                        );
                    }
                    KeyCode::F9 if pressed => {
                        send_action(
                            ctx.action_sender(),
                            RpgAction::LoadCheckpoint,
                        );
                    }
                    KeyCode::Space if pressed && ctx.is_world_active() => {
                        send_action(
                            ctx.action_sender(),
                            RpgAction::Attack,
                        );
                    }
                    KeyCode::KeyW | KeyCode::KeyS | KeyCode::KeyA | KeyCode::KeyD
                        if ctx.is_world_active() || !pressed =>
                    {
                        let move_key = match k.key {
                            KeyCode::KeyW => MoveKey::Forward,
                            KeyCode::KeyS => MoveKey::Backward,
                            KeyCode::KeyA => MoveKey::Left,
                            KeyCode::KeyD => MoveKey::Right,
                            _ => unreachable!(),
                        };
                        send_action(
                            ctx.action_sender(),
                            RpgAction::MoveKey {
                                key: move_key,
                                pressed,
                            },
                        );
                    }
                    _ => {}
                }
            }
            GameInput::MouseButton(m)
                if m.button == MouseButton::Left
                    && m.state == ElementState::Pressed
                    && ctx.is_world_active() =>
            {
                send_action(
                    ctx.action_sender(),
                    RpgAction::Attack,
                );
            }
            GameInput::MouseDelta(delta) if ctx.is_world_active() && delta.x != 0.0 => {
                self.facing_yaw =
                    (self.facing_yaw + delta.x * 0.002).rem_euclid(std::f32::consts::TAU);
                self.signals.facing_yaw.set(self.facing_yaw);
                send_action(
                    ctx.action_sender(),
                    RpgAction::SetFacingYaw(self.facing_yaw),
                );
            }
            _ => {}
        }
    }

    fn on_sync(&mut self, ctx: &mut SyncContext<Self>) {
        if self.signals.save_requested.swap(
            false,
            Ordering::AcqRel,
        ) {
            ctx.pause_world_for_checkpoint();
            let actors = ctx.collect_actor_checkpoints();
            let mut active_timers = Vec::new();
            if let Some(pid) = self.player_id
                && let Some(regen_ms) = ctx.read_actor(
                    pid,
                    |a| a.regen_remaining_ms,
                )
                && regen_ms > 0
            {
                active_timers.push(
                    TimerCheckpoint {
                        kind: String::from("player_regen"),
                        actor_saved_id: Some(0),
                        remaining_ms: regen_ms,
                    },
                );
            }
            let stage = self.signals.quest_stage.read(|s| *s);
            let checkpoint = WorldCheckpoint {
                world_time_ms: ctx.world_time_ms(),
                actors,
                active_timers,
                stage_ids: vec![stage],
            };
            let path = PathBuf::from(CHECKPOINT_PATH);
            match write_checkpoint_file(
                &path,
                &checkpoint,
                &PostcardSaveCodec,
            ) {
                Ok(()) => self.signals.status_banner.set(
                    format!(
                        "Checkpoint saved at {} ms",
                        checkpoint.world_time_ms
                    ),
                ),
                Err(err) => self.signals.status_banner.set(format!("Save error: {err}")),
            }
            ctx.resume_world_from_checkpoint();
        }

        if self.signals.load_requested.swap(
            false,
            Ordering::AcqRel,
        ) {
            ctx.pause_world_for_checkpoint();
            let path = PathBuf::from(CHECKPOINT_PATH);
            match read_checkpoint_file::<WorldCheckpoint<RpgActorData, QuestStage>, _>(
                &path,
                &PostcardSaveCodec,
            ) {
                Ok(checkpoint) => {
                    let ids = ctx.restore_checkpoint_actors(
                        checkpoint.world_time_ms,
                        checkpoint.actors,
                    );
                    let mut new_player = None;
                    let mut new_goblin = None;
                    let mut new_tree = None;
                    for id in ids {
                        if let Some(role) = ctx.read_actor(
                            id,
                            |a| a.role,
                        ) {
                            match role {
                                ActorRole::Player => new_player = Some(id),
                                ActorRole::Goblin => new_goblin = Some(id),
                                ActorRole::CorruptedTree => new_tree = Some(id),
                                _ => {}
                            }
                        }
                    }
                    self.player_id = new_player;
                    self.goblin_id = new_goblin;
                    self.tree_id = new_tree;

                    let stage = checkpoint
                        .stage_ids
                        .first()
                        .copied()
                        .unwrap_or(QuestStage::HuntGoblin);
                    self.signals.quest_stage.set(stage);

                    self.signals.move_dir.set(glam::Vec3::ZERO);
                    if let Some(pid) = new_player {
                        let restored_yaw = ctx
                            .read_actor(
                                pid,
                                |a| a.facing_yaw,
                            )
                            .unwrap_or(0.0);
                        self.facing_yaw = restored_yaw;
                        self.signals.facing_yaw.set(restored_yaw);
                        let sig = self.signals.clone();
                        let rim_effect = self.rim_effect;
                        ctx.attach_actor_script(
                            pid,
                            move |sc| {
                                scripts::player_script(
                                    sc, pid, new_goblin, new_tree, rim_effect, sig,
                                )
                            },
                        );
                        let move_signal = self.signals.move_dir.clone();
                        let yaw_signal = self.signals.facing_yaw.clone();
                        ctx.attach_actor_script(
                            pid,
                            move |sc| {
                                scripts::player_movement_script(
                                    sc,
                                    pid,
                                    move_signal,
                                    yaw_signal,
                                )
                            },
                        );
                        if let Some(regen_ms) = ctx.read_actor(
                            pid,
                            |a| a.regen_remaining_ms,
                        ) && regen_ms > 0
                        {
                            let buff_rim = self.rim_effect;
                            ctx.spawn_buff(
                                pid,
                                BUFF_REGEN_KEY,
                                move |sc| {
                                    scripts::regen_buff_script(
                                        sc, pid, regen_ms, buff_rim,
                                    )
                                },
                            );
                        }
                    }
                    if let (Some(gid), Some(pid)) = (
                        new_goblin, new_player,
                    ) {
                        let sig = self.signals.clone();
                        ctx.attach_actor_script(
                            gid,
                            move |sc| {
                                scripts::goblin_ai_script(
                                    sc, gid, pid, sig,
                                )
                            },
                        );
                    }
                    let q_sig = self.signals.clone();
                    let q_rim = self.rim_effect;
                    ctx.spawn_global_script(
                        move |sc| {
                            scripts::quest_stage_script(
                                sc, new_goblin, new_tree, stage, q_rim, q_sig,
                            )
                        },
                    );
                    ctx.set_flat_mode(FlatMode::InGame);
                    self.signals.status_banner.set(
                        format!(
                            "Loaded checkpoint from world {} ms",
                            checkpoint.world_time_ms
                        ),
                    );
                }
                Err(err) => self.signals.status_banner.set(format!("Load error: {err}")),
            }
            ctx.resume_world_from_checkpoint();
        }
    }

    fn build_snapshot(
        &self,
        world: &WorldReadView<Self>,
    ) -> (
        CameraSnapshot,
        Self::SnapshotData,
    ) {
        let player_pos = self
            .player_id
            .and_then(|id| world.actor_position(id))
            .unwrap_or(
                glam::Vec3::new(
                    0.0, 1.0, 0.0,
                ),
            );

        let forward = facing_forward(self.facing_yaw);
        let camera = CameraSnapshot {
            eye: player_pos - forward * 14.0
                + glam::Vec3::new(
                    0.0, 11.0, 0.0,
                ),
            target: player_pos
                + forward * 1.5
                + glam::Vec3::new(
                    0.0, 0.5, 0.0,
                ),
            up: glam::Vec3::Y,
            fov_y_radians: 48.0_f32.to_radians(),
        };

        let (player_hp, player_max_hp, player_potions, player_regen_ms) = self
            .player_id
            .and_then(
                |id| {
                    world.read_actor(
                        id,
                        |a| {
                            (
                                a.hp,
                                a.max_hp,
                                a.potions,
                                a.regen_remaining_ms,
                            )
                        },
                    )
                },
            )
            .unwrap_or(
                (
                    0, 100, 0, 0,
                ),
            );

        let goblin_hp = self.goblin_id.and_then(
            |id| {
                world.read_actor(
                    id,
                    |a| a.hp,
                )
            },
        );
        let tree_hp = self.tree_id.and_then(
            |id| {
                world.read_actor(
                    id,
                    |a| a.hp,
                )
            },
        );

        let snap = RpgSnapshot {
            flat_mode: *world.modes().flat(),
            stack_modes: world.modes().stack_modes(),
            player_hp,
            player_max_hp,
            player_potions,
            player_regen_ms,
            goblin_hp,
            tree_hp,
            quest_stage: self.signals.quest_stage.read(|s| *s),
            status_banner: self.signals.status_banner.read(|s| s.clone()),
        };

        (
            camera, snap,
        )
    }

    fn build_gizmos(&self, world: &WorldReadView<Self>) -> Vec<GizmoBox> {
        let Some(player_pos) = self.player_id.and_then(|id| world.actor_position(id)) else {
            return Vec::new();
        };

        let yaw = self.facing_yaw;
        let forward = facing_forward(yaw);
        let rot = facing_rotation(yaw);
        let ground_y = (player_pos.y - 0.78).max(0.08);
        let base = glam::Vec3::new(
            player_pos.x,
            ground_y,
            player_pos.z,
        );

        let mut gizmos = Vec::with_capacity(5);
        // Visor bar mounted on the knight's front face.
        gizmos.push(
            GizmoBox {
                position: player_pos
                    + forward * 0.58
                    + glam::Vec3::new(
                        0.0, 0.45, 0.0,
                    ),
                rotation: rot,
                half_extents: glam::Vec3::new(
                    0.36, 0.14, 0.08,
                ),
                color: [0.95, 0.98, 1.0],
            },
        );
        // Center forward aim ray matching the 3.2m attack range.
        gizmos.push(
            GizmoBox {
                position: base + forward * 1.6,
                rotation: rot,
                half_extents: glam::Vec3::new(
                    0.045, 0.035, 1.6,
                ),
                color: [1.0, 0.92, 0.2],
            },
        );
        // Forward tip marker at max attack reach.
        gizmos.push(
            GizmoBox {
                position: base + forward * 3.2,
                rotation: rot,
                half_extents: glam::Vec3::new(
                    0.14, 0.06, 0.14,
                ),
                color: [1.0, 0.95, 0.35],
            },
        );

        // +-15 degree effective attack cone boundaries.
        let cone_half = 15.0_f32.to_radians();
        for offset in [-cone_half, cone_half] {
            let edge_yaw = yaw + offset;
            let edge_dir = facing_forward(edge_yaw);
            gizmos.push(
                GizmoBox {
                    position: base + edge_dir * 1.6,
                    rotation: facing_rotation(edge_yaw),
                    half_extents: glam::Vec3::new(
                        0.03, 0.03, 1.6,
                    ),
                    color: [1.0, 0.45, 0.15],
                },
            );
        }

        gizmos
    }

    fn wants_cursor_grab(&self, world: &WorldReadView<Self>) -> bool {
        world.modes().is_world_active()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn facing_vectors_and_relative_movement_support_full_360_degrees() {
        let right_turn = std::f32::consts::FRAC_PI_2;
        let fwd = facing_forward(right_turn);
        assert!((fwd - glam::Vec3::X).length() < 1e-5);

        let world_w = local_move_to_world(
            glam::Vec3::NEG_Z,
            right_turn,
        );
        assert!((world_w - glam::Vec3::X).length() < 1e-5);

        let world_d = local_move_to_world(
            glam::Vec3::X,
            right_turn,
        );
        assert!((world_d - glam::Vec3::Z).length() < 1e-5);

        let rot_fwd = facing_rotation(right_turn) * glam::Vec3::NEG_Z;
        assert!((rot_fwd - fwd).length() < 1e-5);

        let full_turn = std::f32::consts::PI * 1.5;
        let left_fwd = facing_forward(full_turn);
        assert!((left_fwd - glam::Vec3::NEG_X).length() < 1e-5);
    }
}
