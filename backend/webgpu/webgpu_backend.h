#ifndef WEBGPU_BACKEND_H
#define WEBGPU_BACKEND_H

#include "../../core/include/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

// Initialize WebGPU backend
// Returns 0 on success, -1 if WebGPU unavailable
int webgpu_init(void);

// Cleanup WebGPU backend
void webgpu_cleanup(void);

// Check if WebGPU is available and initialized
int webgpu_available(void);

// Register all WebGPU-accelerated operations
void webgpu_register_ops(void);

// Internal use - get WebGPU context (opaque pointers)
struct WGPUDeviceImpl* webgpu_get_device_internal(void);
struct WGPUQueueImpl* webgpu_get_queue_internal(void);

#ifdef __cplusplus
}
#endif

#endif // WEBGPU_BACKEND_H
