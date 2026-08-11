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
#define MTT_PLUGIN_STATUS_NOT_READY INT32_C(7)
#define MTT_PLUGIN_STATUS_CLASS_TABLE_CONFLICT INT32_C(8)

#define MTT_MAIN_DEVICE_POSITION_CLASS_COUNT_V1 UINT32_C(21)
#define MTT_MAIN_DEVICE_POSITION_K_COUNT_V1 UINT32_C(249)
#define MTT_MAIN_DEVICE_POSITION_FUNCTION_NAME_CAPACITY UINT32_C(256)
#define MTT_MAIN_DEVICE_POSITION_LAYER_COUNT_V1 UINT32_C(12)

/*
 * A zero execution status is success. Non-zero values are positive INT32 and
 * preserve the first failure through all twelve decoder layers and through
 * the A/B recurrent step state.
 */
#define MTT_MAIN_DEVICE_POSITION_STATUS_CATEGORY_SHIFT UINT32_C(28)
#define MTT_MAIN_DEVICE_POSITION_STATUS_LAYER_SHIFT UINT32_C(24)
#define MTT_MAIN_DEVICE_POSITION_STATUS_OPERATION_SHIFT UINT32_C(22)
#define MTT_MAIN_DEVICE_POSITION_STATUS_DETAIL_MASK UINT32_C(0x003fffff)
#define MTT_MAIN_DEVICE_POSITION_STATUS_CATEGORY_MASK UINT32_C(0x7)
#define MTT_MAIN_DEVICE_POSITION_STATUS_LAYER_MASK UINT32_C(0xf)
#define MTT_MAIN_DEVICE_POSITION_STATUS_OPERATION_MASK UINT32_C(0x3)

#define MTT_MAIN_DEVICE_POSITION_STATUS_INVALID_K UINT32_C(1)
#define MTT_MAIN_DEVICE_POSITION_STATUS_CUDA_GRAPH_UPDATE UINT32_C(2)

#define MTT_MAIN_DEVICE_POSITION_STATUS_OPERATION_SELECTOR UINT32_C(0)
#define MTT_MAIN_DEVICE_POSITION_STATUS_OPERATION_QK UINT32_C(1)
#define MTT_MAIN_DEVICE_POSITION_STATUS_OPERATION_PV UINT32_C(2)

#define MTT_MAIN_DEVICE_POSITION_OPERATION_QK UINT32_C(1)
#define MTT_MAIN_DEVICE_POSITION_OPERATION_PV UINT32_C(2)
#define MTT_MAIN_DEVICE_POSITION_PARAMETER_KERNEL_PARAMS UINT32_C(1)
#define MTT_MAIN_DEVICE_POSITION_PARAMETER_EXTRA UINT32_C(2)

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

typedef struct mtt_main_device_position_class_v1 {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t operation;
  uint32_t class_index;
  uint32_t parameter_transport;
  uint32_t block_x;
  uint32_t block_y;
  uint32_t block_z;
  uint32_t shared_memory_bytes;
  uint32_t reserved_0;
  uint64_t parameter_offset;
  uint64_t parameter_size;
  char function_name[MTT_MAIN_DEVICE_POSITION_FUNCTION_NAME_CAPACITY];
  uint64_t reserved[4];
} mtt_main_device_position_class_v1_t;

typedef struct mtt_main_device_position_k_v1 {
  uint32_t struct_size;
  uint32_t abi_version;
  int32_t active_k;
  int32_t qk_class_index;
  uint32_t qk_grid_x;
  uint32_t qk_grid_y;
  uint32_t qk_grid_z;
  int32_t pv_class_index;
  uint32_t pv_grid_x;
  uint32_t pv_grid_y;
  uint32_t pv_grid_z;
  uint32_t reserved_0;
  uint64_t reserved[4];
} mtt_main_device_position_k_v1_t;

typedef struct mtt_main_device_position_class_table_v1 {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t class_count;
  uint32_t k_count;
  mtt_main_device_position_class_v1_t
      classes[MTT_MAIN_DEVICE_POSITION_CLASS_COUNT_V1];
  mtt_main_device_position_k_v1_t
      k_records[MTT_MAIN_DEVICE_POSITION_K_COUNT_V1];
  uint64_t reserved[4];
} mtt_main_device_position_class_table_v1_t;

/*
 * The caller initializes struct_size and abi_version before this call.
 * The returned pointers and fixed-size creator descriptors remain valid until
 * the plugin shared object is unloaded.
 */
MTT_PLUGIN_API mtt_plugin_status_t
mtt_plugin_get_api_v1(mtt_plugin_api_v1_t* api);

/*
 * Available only after the first mode-8 bank has completed discovery. The
 * table contains no process address and no opaque cuBLAS parameter bytes.
 * Every later bank build must reproduce the same canonical table or fail.
 */
MTT_PLUGIN_API mtt_plugin_status_t
mtt_plugin_get_main_device_position_class_table_v1(
    mtt_main_device_position_class_table_v1_t* table);

#ifdef __cplusplus
}
#endif

#endif
