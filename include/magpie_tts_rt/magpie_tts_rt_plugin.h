#ifndef MAGPIE_TTS_RT_MAGPIE_TTS_RT_PLUGIN_H_
#define MAGPIE_TTS_RT_MAGPIE_TTS_RT_PLUGIN_H_

#include <stdint.h>

#if defined(_WIN32)
#if defined(MAGPIE_TTS_RT_BUILDING_PLUGIN_LIBRARY)
#define MTT_PLUGIN_API __declspec(dllexport)
#else
#define MTT_PLUGIN_API __declspec(dllimport)
#endif
#elif defined(__GNUC__) || defined(__clang__)
#define MTT_PLUGIN_API __attribute__((visibility("default")))
#else
#define MTT_PLUGIN_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define MTT_PLUGIN_ABI_VERSION_1 UINT32_C(1)
#define MTT_PLUGIN_CREATOR_COUNT_V1 UINT32_C(5)
#define MTT_PLUGIN_CREATOR_NAME_CAPACITY UINT32_C(64)
#define MTT_PLUGIN_CREATOR_VERSION_CAPACITY UINT32_C(16)
#define MTT_PLUGIN_NAMESPACE_CAPACITY UINT32_C(64)

typedef int32_t mtt_plugin_status_t;

#define MTT_PLUGIN_STATUS_OK INT32_C(0)
#define MTT_PLUGIN_STATUS_ALREADY_REGISTERED INT32_C(1)
#define MTT_PLUGIN_STATUS_INVALID_ARGUMENT INT32_C(2)
#define MTT_PLUGIN_STATUS_ABI_MISMATCH INT32_C(3)
#define MTT_PLUGIN_STATUS_REGISTRY_UNAVAILABLE INT32_C(4)
#define MTT_PLUGIN_STATUS_REGISTRATION_CONFLICT INT32_C(5)
#define MTT_PLUGIN_STATUS_REGISTRATION_FAILED INT32_C(6)

typedef struct mtt_plugin_creator_v1 {
  uint32_t struct_size;
  uint32_t abi_version;
  char name[MTT_PLUGIN_CREATOR_NAME_CAPACITY];
  char version[MTT_PLUGIN_CREATOR_VERSION_CAPACITY];
  char plugin_namespace[MTT_PLUGIN_NAMESPACE_CAPACITY];
} mtt_plugin_creator_v1_t;

typedef mtt_plugin_status_t (*mtt_plugin_register_v1_fn)(void);

typedef struct mtt_plugin_api_v1 {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t creator_count;
  uint32_t reserved_0;
  mtt_plugin_creator_v1_t creators[MTT_PLUGIN_CREATOR_COUNT_V1];
  mtt_plugin_register_v1_fn register_plugins;
  uint64_t reserved[4];
} mtt_plugin_api_v1_t;

/*
 * The caller initializes struct_size and abi_version before this call.
 * The returned pointers and fixed-size creator descriptors remain valid until
 * the plugin shared object is unloaded.
 */
MTT_PLUGIN_API mtt_plugin_status_t
mtt_plugin_get_api_v1(mtt_plugin_api_v1_t* api);

#ifdef __cplusplus
}
#endif

#endif
