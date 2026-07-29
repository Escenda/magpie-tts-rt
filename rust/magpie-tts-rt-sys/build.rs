use std::env;
use std::path::{Path, PathBuf};

const LIB_DIR_ENV: &str = "MAGPIE_TTS_RT_LIB_DIR";
const HEADER: &str = "../../include/magpie_tts_rt/magpie_tts_rt.h";

fn native_library_filename(target_os: &str) -> &'static str {
    match target_os {
        "linux" => "libmagpie_tts_rt.so",
        unsupported => panic!("native-link is unsupported for target OS {unsupported}"),
    }
}

fn require_native_library(lib_dir: &Path, target_os: &str) -> PathBuf {
    let library = lib_dir.join(native_library_filename(target_os));
    assert!(
        library.is_file(),
        "{} does not contain the required native library {}",
        lib_dir.display(),
        library.display()
    );
    library
}

fn main() {
    println!("cargo:rerun-if-env-changed={LIB_DIR_ENV}");
    println!("cargo:rerun-if-changed={HEADER}");

    if env::var_os("CARGO_FEATURE_NATIVE_LINK").is_none() {
        return;
    }

    let target_os = env::var("CARGO_CFG_TARGET_OS")
        .expect("Cargo must provide CARGO_CFG_TARGET_OS when running build.rs");
    let configured_dir = env::var_os(LIB_DIR_ENV).unwrap_or_else(|| {
        panic!(
            "feature `native-link` requires an explicit absolute {LIB_DIR_ENV}; \
             implicit system-library discovery is intentionally disabled"
        )
    });
    let lib_dir = PathBuf::from(configured_dir);
    assert!(
        lib_dir.is_absolute(),
        "{LIB_DIR_ENV} must be an absolute path, got {}",
        lib_dir.display()
    );

    let library = require_native_library(&lib_dir, &target_os);
    println!("cargo:rerun-if-changed={}", library.display());
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=dylib=magpie_tts_rt");
}
