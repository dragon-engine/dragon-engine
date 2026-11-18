use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::effect::EffectAttachment;
use crate::physics::BodyKind;

/// Pluggable serialization codec trait for save/load checkpoints (for example `postcard` or `ron`).
pub trait SaveCodec<T> {
    type Error: std::fmt::Display;

    fn encode(&self, value: &T) -> Result<Vec<u8>, Self::Error>;
    fn decode(&self, bytes: &[u8]) -> Result<T, Self::Error>;
}

/// Serializable actor record used during Checkpoint State Reconstruction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActorCheckpoint<A> {
    pub saved_id: usize,
    pub position: [f32; 3],
    pub rotation: [f32; 4],
    pub half_extents: [f32; 3],
    pub color: [f32; 3],
    #[serde(default)]
    pub effect: Option<EffectAttachment>,
    pub body_kind: BodyKind,
    pub mass: f32,
    pub data: A,
}

/// Serializable remaining timer entry reconstructed on checkpoint load.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimerCheckpoint {
    pub kind: String,
    pub actor_saved_id: Option<usize>,
    pub remaining_ms: u64,
}

/// Complete serializable checkpoint snapshot produced while `ClockDomain::World` is paused.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorldCheckpoint<A, Q> {
    pub world_time_ms: u64,
    pub actors: Vec<ActorCheckpoint<A>>,
    pub active_timers: Vec<TimerCheckpoint>,
    pub stage_ids: Vec<Q>,
}

/// Writes a checkpoint to disk synchronously outside async gameplay futures.
pub fn write_checkpoint_file<T, C: SaveCodec<T>>(
    path: &Path,
    checkpoint: &T,
    codec: &C,
) -> Result<(), String> {
    let bytes = codec
        .encode(checkpoint)
        .map_err(|err| format!("failed to encode checkpoint: {err}"))?;
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)
            .map_err(|err| format!("failed to create save directory: {err}"))?;
    }
    fs::write(
        path, bytes,
    )
    .map_err(|err| format!("failed to write checkpoint file: {err}"))
}

/// Reads a checkpoint from disk synchronously outside async gameplay futures.
pub fn read_checkpoint_file<T, C: SaveCodec<T>>(path: &Path, codec: &C) -> Result<T, String> {
    let bytes = fs::read(path).map_err(
        |err| {
            format!(
                "failed to read checkpoint file {}: {err}",
                path.display()
            )
        },
    )?;
    codec.decode(&bytes).map_err(
        |err| {
            format!(
                "failed to decode checkpoint file {}: {err}",
                path.display()
            )
        },
    )
}
