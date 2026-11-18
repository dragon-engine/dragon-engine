mod app;
mod save_codec;
mod scripts;
mod ui;

use engine::{EffectRegistry, EngineConfig, EngineRunner};
use fx_fresnel_rim::FresnelRimEffect;

use crate::app::RpgGame;
use crate::ui::RpgUi;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    engine::init_tracing();
    let config = EngineConfig {
        title: String::from("Goblin Meadow"),
        ..Default::default()
    };
    let mut effects = EffectRegistry::new();
    let rim_effect = effects.register(FresnelRimEffect::new());
    EngineRunner::run_with_effects(
        config,
        effects,
        RpgGame::new(rim_effect),
        RpgUi::new(),
    )?;
    Ok(())
}
