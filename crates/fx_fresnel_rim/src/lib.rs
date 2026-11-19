use engine::{EffectAttachment, EffectBlendMode, EffectId, EffectShaderSpec, RenderEffect};

/// Typed configuration helper that packs Fresnel rim parameters into `EffectAttachment`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FresnelRimParams {
    pub color: [f32; 3],
    pub intensity: f32,
    pub power: f32,
    pub pulse_speed: f32,
    pub edge_boost: f32,
}

impl FresnelRimParams {
    #[must_use]
    pub fn new(color: [f32; 3], intensity: f32, power: f32) -> Self {
        Self {
            color,
            intensity,
            power,
            pulse_speed: 0.0,
            edge_boost: 0.55,
        }
    }

    #[must_use]
    pub fn with_pulse(mut self, pulse_speed: f32) -> Self {
        self.pulse_speed = pulse_speed;
        self
    }

    #[must_use]
    pub fn with_edge_boost(mut self, edge_boost: f32) -> Self {
        self.edge_boost = edge_boost;
        self
    }

    #[must_use]
    pub fn attach(self, effect: EffectId) -> EffectAttachment {
        EffectAttachment {
            effect,
            params: [
                [self.color[0], self.color[1], self.color[2], self.intensity],
                [self.power, self.pulse_speed, self.edge_boost, 0.0],
            ],
        }
    }
}

/// Pluggable Fresnel Rim overlay effect implemented with `rust-gpu` SPIR-V.
#[derive(Debug, Default, Clone, Copy)]
pub struct FresnelRimEffect;

impl FresnelRimEffect {
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl RenderEffect for FresnelRimEffect {
    fn shader_spec(&self) -> EffectShaderSpec {
        EffectShaderSpec {
            label: "Fresnel Rim Shader",
            spirv_bytes: include_bytes!(env!("FRESNEL_RIM_SHADER_SPV")),
            vertex_entry: "vs_main",
            fragment_entry: "fs_main",
            blend: EffectBlendMode::AlphaBlend,
        }
    }
}

#[cfg(test)]
#[allow(clippy::disallowed_methods)]
mod tests {
    use super::*;

    #[test]
    fn fresnel_rim_params_pack_into_effect_attachment() {
        let mut reg = engine::EffectRegistry::new();
        let fx_id = reg.register(FresnelRimEffect::new());
        let attachment = FresnelRimParams::new(
            [0.2, 0.8, 1.0],
            0.9,
            2.5,
        )
        .with_pulse(4.0)
        .with_edge_boost(0.35)
        .attach(fx_id);

        assert_eq!(
            attachment.effect,
            fx_id
        );
        assert_eq!(
            attachment.params[0],
            [0.2, 0.8, 1.0, 0.9]
        );
        assert_eq!(
            attachment.params[1],
            [2.5, 4.0, 0.35, 0.0]
        );
    }

    #[test]
    fn fresnel_rim_shader_emits_valid_spirv_and_compiles_on_vulkan() {
        let spec = FresnelRimEffect::new().shader_spec();
        assert!(spec.spirv_bytes.len() >= 20);
        assert_eq!(
            spec.spirv_bytes.len() % 4,
            0
        );
        let magic = u32::from_le_bytes(
            [
                spec.spirv_bytes[0],
                spec.spirv_bytes[1],
                spec.spirv_bytes[2],
                spec.spirv_bytes[3],
            ],
        );
        assert_eq!(
            magic,
            0x0723_0203
        );
        engine::validate_effect_shader_vulkan(spec)
            .expect("FresnelRimEffect SPIR-V must compile into a Vulkan 1.3 pipeline");
    }
}
