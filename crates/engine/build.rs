use std::error::Error;
use std::path::PathBuf;

use spirv_builder::SpirvBuilder;

fn main() -> Result<(), Box<dyn Error>> {
    println!("cargo:rerun-if-changed=build.rs");
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let shader_dir = manifest_dir.join("shaders/core_shader");
    let out_dir =
        PathBuf::from(std::env::var_os("OUT_DIR").ok_or("OUT_DIR environment variable missing")?);

    // TODO(wgpu): `wgpu-core 30.0.1` rejects DXIL passthrough shader modules when
    // `entry_points.len() != 1` (`IncorrectPassthroughEntryPointCount`). When adding Phase 7 DX12
    // passthrough packaging, emit one DXIL module per entry point instead of one multi-entry blob:
    // https://github.com/gfx-rs/wgpu/issues/9991
    let mut builder = SpirvBuilder::new(
        shader_dir,
        "spirv-unknown-vulkan1.1",
    )
    .target_dir_path(out_dir.join("spirv-builder"));
    builder.build_script.defaults = true;
    let result = builder.build()?;

    println!(
        "cargo:rustc-env=CORE_SHADER_SPV={}",
        result.module.unwrap_single().display()
    );
    Ok(())
}
