#pragma once

#include "manifest/manifest.hpp"
#include "runtime/session_resources.hpp"
#include "runtime/startup_golden.hpp"

namespace magpie_tts_rt {

// Executes the authenticated prepared-token fixture through the same complete
// Text Encoder -> Decoder -> Local AR -> NanoCodec path used by requests.
// Audio is drained only to exercise bounded backpressure; no bytes escape the
// session until all counts and hashes match.
void run_startup_golden_gate(
    const RuntimeBundleManifest& manifest,
    const StartupGoldenFixture& fixture,
    SessionResources& resources);

}  // namespace magpie_tts_rt
