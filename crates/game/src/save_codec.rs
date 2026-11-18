use engine::SaveCodec;
use serde::Serialize;
use serde::de::DeserializeOwned;

/// Binary `postcard` codec implementing `engine::SaveCodec` for checkpoint persistence.
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct PostcardSaveCodec;

impl<T> SaveCodec<T> for PostcardSaveCodec
where
    T: Serialize + DeserializeOwned,
{
    type Error = postcard::Error;

    fn encode(&self, value: &T) -> Result<Vec<u8>, Self::Error> {
        postcard::to_allocvec(value)
    }

    fn decode(&self, bytes: &[u8]) -> Result<T, Self::Error> {
        postcard::from_bytes(bytes)
    }
}

#[cfg(test)]
#[allow(clippy::disallowed_methods)]
mod tests {
    use engine::{ActorCheckpoint, BodyKind, SaveCodec, WorldCheckpoint};

    use super::*;
    use crate::app::{ActorRole, QuestStage, RpgActorData};

    #[test]
    fn postcard_checkpoint_roundtrip() {
        let cp = WorldCheckpoint {
            world_time_ms: 4200,
            actors: vec![ActorCheckpoint {
                saved_id: 0,
                position: [1.0, 2.0, 3.0],
                rotation: [0.0, 0.0, 0.0, 1.0],
                half_extents: [0.5, 0.9, 0.5],
                color: [0.2, 0.6, 0.9],
                effect: None,
                body_kind: BodyKind::Dynamic,
                mass: 2.0,
                data: RpgActorData {
                    name: String::from("Knight"),
                    role: ActorRole::Player,
                    hp: 85,
                    max_hp: 100,
                    potions: 2,
                    attack_power: 18,
                    regen_remaining_ms: 3000,
                    facing_yaw: 1.25,
                },
            }],
            active_timers: vec![],
            stage_ids: vec![QuestStage::PurifyTree],
        };

        let bytes = PostcardSaveCodec.encode(&cp).expect("encode");
        let decoded: WorldCheckpoint<RpgActorData, QuestStage> =
            PostcardSaveCodec.decode(&bytes).expect("decode");
        assert_eq!(
            decoded.world_time_ms,
            4200
        );
        assert_eq!(
            decoded.stage_ids,
            vec![QuestStage::PurifyTree]
        );
        assert_eq!(
            decoded.actors[0].data.hp,
            85
        );
        assert_eq!(
            decoded.actors[0].data.regen_remaining_ms,
            3000
        );
        assert!((decoded.actors[0].data.facing_yaw - 1.25).abs() < 1e-6);
    }
}
