use std::error::Error;
use std::path::PathBuf;

use spirv_builder::SpirvBuilder;

fn main() -> Result<(), Box<dyn Error>> {
    println!("cargo:rerun-if-changed=build.rs");
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let shader_dir = manifest_dir.join("shaders/fresnel_rim_shader");
    let out_dir =
        PathBuf::from(std::env::var_os("OUT_DIR").ok_or("OUT_DIR environment variable missing")?);

    let mut builder = SpirvBuilder::new(
        shader_dir,
        "spirv-unknown-vulkan1.1",
    )
    .target_dir_path(out_dir.join("spirv-builder"));
    builder.build_script.defaults = true;
    let result = builder.build()?;

    println!(
        "cargo:rustc-env=FRESNEL_RIM_SHADER_SPV={}",
        result.module.unwrap_single().display()
    );
    Ok(())
}
