#![no_std]

use shader_abi::{SceneUniforms, Vec4f};
use spirv_std::glam::{Mat4, Vec2, Vec3, Vec4};
use spirv_std::num_traits::Float;
use spirv_std::spirv;

#[repr(C)]
#[derive(Copy, Clone)]
pub struct RimUniform {
    /// `(r, g, b, intensity)`
    pub color_intensity: Vec4f,
    /// `(power, pulse_speed, edge_boost, 0.0)`
    pub power_pulse: Vec4f,
}

#[spirv(vertex)]
pub fn vs_main(
    pos: Vec3,
    normal: Vec3,
    uv: Vec2,
    _color: Vec3,
    #[spirv(descriptor_set = 0, binding = 0, uniform)] scene: &SceneUniforms,
    #[spirv(position)] out_pos: &mut Vec4,
    out_world_pos: &mut Vec3,
    out_normal: &mut Vec3,
    out_uv: &mut Vec2,
) {
    let view_proj = Mat4::from_cols_array_2d(&scene.view_proj.to_cols_array_2d());
    *out_pos = view_proj
        * Vec4::from(
            (
                pos, 1.0,
            ),
        );
    *out_world_pos = pos;
    *out_normal = normal;
    *out_uv = uv;
}

fn safe_normalize(v: Vec3) -> Vec3 {
    let len_sq = v.length_squared();
    if len_sq > 1e-8 {
        v / len_sq.sqrt()
    } else {
        Vec3::ZERO
    }
}

#[spirv(fragment)]
pub fn fs_main(
    in_world_pos: Vec3,
    in_normal: Vec3,
    in_uv: Vec2,
    #[spirv(descriptor_set = 0, binding = 0, uniform)] scene: &SceneUniforms,
    #[spirv(descriptor_set = 1, binding = 0, uniform)] rim: &RimUniform,
    output: &mut Vec4,
) {
    let n = safe_normalize(in_normal);
    let eye = Vec3::new(
        scene.camera_eye_time.x,
        scene.camera_eye_time.y,
        scene.camera_eye_time.z,
    );
    let v = safe_normalize(eye - in_world_pos);
    let ndotv = n.dot(v).clamp(
        0.0, 1.0,
    );
    let power = rim.power_pulse.x.max(0.25);
    let fresnel = (1.0 - ndotv).powf(power);

    let edge_boost = rim.power_pulse.z.max(0.0);
    let edge_x = in_uv.x.min(1.0 - in_uv.x);
    let edge_y = in_uv.y.min(1.0 - in_uv.y);
    let edge_dist = edge_x.min(edge_y);
    let edge_factor = (1.0
        - (edge_dist * 10.0).clamp(
            0.0, 1.0,
        ))
    .powf(2.0);
    let rim_mask = (fresnel + edge_factor * edge_boost).clamp(
        0.0, 1.0,
    );

    let pulse_speed = rim.power_pulse.y;
    let pulse = if pulse_speed > 0.0 {
        let world_time = scene.camera_eye_time.w;
        0.70 + 0.30 * (world_time * pulse_speed).sin()
    } else {
        1.0
    };
    let intensity = rim.color_intensity.w.max(0.0);
    let alpha = (rim_mask * intensity * pulse).clamp(
        0.0, 1.0,
    );

    *output = Vec4::new(
        rim.color_intensity.x,
        rim.color_intensity.y,
        rim.color_intensity.z,
        alpha,
    );
}
