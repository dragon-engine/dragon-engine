#![no_std]

// TODO(wgpu): `wgpu-hal 30.0.1` does not enable Vulkan `runtime_descriptor_array`, so unsized
// `spirv_std::RuntimeArray<Image!(...)>` descriptor bindings fail Vulkan validation under
// `PASSTHROUGH_SHADERS`. Use a fixed-size `[Image!(...); N]` array matching
// `BindGroupLayoutEntry::count` behind one centralized texture-sampling helper:
// https://github.com/gfx-rs/wgpu/issues/7347
// https://github.com/gfx-rs/wgpu/issues/3637

use shader_abi::SceneUniforms;
use spirv_std::glam::{Mat4, Vec2, Vec3, Vec4};
use spirv_std::image::Image2d;
use spirv_std::{Sampler, spirv};

#[spirv(vertex)]
pub fn vs_main(
    pos: Vec3,
    _normal: Vec3,
    _uv: Vec2,
    color: Vec3,
    #[spirv(descriptor_set = 0, binding = 0, uniform)] scene: &SceneUniforms,
    #[spirv(position)] out_pos: &mut Vec4,
    out_color: &mut Vec3,
) {
    let view_proj = Mat4::from_cols_array_2d(&scene.view_proj.to_cols_array_2d());
    *out_pos = view_proj
        * Vec4::from(
            (
                pos, 1.0,
            ),
        );
    *out_color = color;
}

#[spirv(fragment)]
pub fn fs_main(in_color: Vec3, output: &mut Vec4) {
    *output = Vec4::from(
        (
            in_color, 1.0,
        ),
    );
}

#[spirv(vertex)]
pub fn ui_vs_main(
    #[spirv(vertex_index)] vid: u32,
    #[spirv(position)] out_pos: &mut Vec4,
    out_uv: &mut Vec2,
) {
    let x = ((vid & 1) as f32) * 4.0 - 1.0;
    let y = ((vid >> 1) as f32) * 4.0 - 1.0;
    let u = ((vid & 1) as f32) * 2.0;
    let v = 1.0 - ((vid >> 1) as f32) * 2.0;
    *out_pos = Vec4::new(
        x, y, 0.0, 1.0,
    );
    *out_uv = Vec2::new(
        u, v,
    );
}

#[spirv(fragment)]
pub fn ui_fs_main(
    in_uv: Vec2,
    #[spirv(descriptor_set = 0, binding = 0)] ui_tex: &Image2d,
    #[spirv(descriptor_set = 0, binding = 1)] ui_sampler: &Sampler,
    output: &mut Vec4,
) {
    *output = ui_tex.sample(
        *ui_sampler,
        in_uv,
    );
}
