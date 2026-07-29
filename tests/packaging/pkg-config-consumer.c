#include <stdint.h>

#include <magpie_tts_rt/magpie_tts_rt.h>

int main(void) {
  mtt_api_v1_t api = {
      .struct_size = sizeof(api),
      .abi_version = MTT_ABI_VERSION_1,
  };
  return mtt_get_api(MTT_ABI_VERSION_1, &api) == MTT_STATUS_OK ? 0 : 1;
}
