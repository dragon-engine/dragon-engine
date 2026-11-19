use std::ops::Range;

use serde::{Deserialize, Serialize};

/// Opaque handle identifying a registered `RenderEffect` implementation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EffectId(pub(crate) u32);

impl EffectId {
    #[must_use]
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

/// Game-agnostic effect attachment stored per actor mesh in `engine` SoA storage.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct EffectAttachment {
    pub effect: EffectId,
    pub params: [[f32; 4]; 2],
}

/// Blending behavior applied when `engine` composites a `RenderEffect` pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EffectBlendMode {
    Opaque,
    AlphaBlend,
    Additive,
}

/// Declarative SPIR-V shader specification supplied by a pluggable `RenderEffect`.
/// Keeps graphics API handles and `unsafe` code private inside `crates/engine`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EffectShaderSpec {
    pub label: &'static str,
    pub spirv_bytes: &'static [u8],
    pub vertex_entry: &'static str,
    pub fragment_entry: &'static str,
    pub blend: EffectBlendMode,
}

/// One mesh draw range with its attached opaque effect parameters.
#[derive(Debug, Clone)]
pub(crate) struct EffectDrawItem {
    pub(crate) vertex_range: Range<u32>,
    pub(crate) params: [[f32; 4]; 2],
}

/// Pluggable 3D overlay effect interface implemented by standard or external effect crates.
pub trait RenderEffect: Send + 'static {
    /// Returns the compiled `rust-gpu` SPIR-V module and pipeline state for this effect.
    fn shader_spec(&self) -> EffectShaderSpec;
}

/// Registry passed to `EngineRunner::run_with_effects` to attach workspace or third-party effects.
#[derive(Default)]
pub struct EffectRegistry {
    effects: Vec<Box<dyn RenderEffect>>,
}

impl EffectRegistry {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register<E: RenderEffect>(&mut self, effect: E) -> EffectId {
        let id = u32::try_from(self.effects.len())
            .unwrap_or_else(|err| panic!("Too many registered render effects: {err}"));
        self.effects.push(Box::new(effect));
        EffectId(id)
    }

    pub(crate) fn into_effects(self) -> Vec<Box<dyn RenderEffect>> {
        self.effects
    }
}
