#![no_std]
#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals)]
#![doc = "Raw bindings for the MagpieTTS-RT C ABI.

The optional `native-link` feature requires the absolute
`MAGPIE_TTS_RT_LIB_DIR` build environment variable. Without that feature this
crate compiles and tests its ABI declarations without linking a native
library."]

include!("bindings.rs");
include!("constants.rs");
