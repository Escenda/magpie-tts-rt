#include <stddef.h>
#include <stdint.h>

#include "magpie_tts_rt/magpie_tts_rt.h"

_Static_assert(sizeof(mtt_status_t) == sizeof(int32_t), "status must be int32");
_Static_assert(sizeof(mtt_error_stage_t) == sizeof(int32_t), "error stage must be int32");
_Static_assert(offsetof(mtt_error_v1_t, struct_size) == 0, "header must begin at offset zero");
_Static_assert(
    sizeof(((mtt_model_desc_v1_t*)0)->expected_manifest_sha256) ==
        MTT_SHA256_BYTES,
    "model trust anchor must be an exact SHA-256 digest");
_Static_assert(
    sizeof(((mtt_model_info_v1_t*)0)->tokenizer_identity_sha256) ==
        MTT_SHA256_BYTES,
    "model tokenizer identity must be an exact SHA-256 digest");
_Static_assert(
    sizeof(mtt_model_info_v1_t) == 136,
    "model info ABI v1 layout changed");
_Static_assert(
    offsetof(mtt_alignment_event_v1_t, struct_size) == 0,
    "alignment event must begin with an ABI header");
_Static_assert(
    offsetof(mtt_audio_lease_v1_t, alignment_events) == 72,
    "audio lease alignment event pointer layout changed");
_Static_assert(offsetof(mtt_api_v1_t, struct_size) == 0, "API must begin at offset zero");

int main(void) {
  mtt_api_v1_t api = {0};
  api.struct_size = (uint32_t)sizeof(api);
  api.abi_version = MTT_ABI_VERSION_1;
  if (mtt_get_api(MTT_ABI_VERSION_1, &api) != MTT_STATUS_OK) {
    return 1;
  }
  return api.runtime_create == NULL || api.model_get_info == NULL ||
         api.audio_release == NULL;
}
