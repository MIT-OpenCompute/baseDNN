#include "webgpu_backend.h"
#include <webgpu/webgpu.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

// ====================================================
// WebGPU Context (Singleton)
// ====================================================

typedef struct {
    WGPUInstance instance;
    WGPUAdapter adapter;
    WGPUDevice device;
    WGPUQueue queue;
    int initialized;
} WebGPUContext;

static WebGPUContext g_webgpu_ctx = {NULL, NULL, NULL, NULL, 0};

// ====================================================
// Callback Handlers
// ====================================================

typedef struct {
    WGPUAdapter adapter;
    volatile int done;
} AdapterRequestData;

typedef struct {
    WGPUDevice device;
    volatile int done;
} DeviceRequestData;

static void adapter_request_callback(WGPURequestAdapterStatus status,
                                     WGPUAdapter adapter,
                                     WGPUStringView message,
                                     void * userdata1,
                                     void * userdata2) {
    (void)userdata2;
    AdapterRequestData* data = (AdapterRequestData*)userdata1;
    if (status == WGPURequestAdapterStatus_Success) {
        data->adapter = adapter;
    } else {
        fprintf(stderr, "WebGPU adapter request failed\n");
    }
    data->done = 1;
}

static void device_request_callback(WGPURequestDeviceStatus status,
                                   WGPUDevice device,
                                   WGPUStringView message,
                                   void * userdata1,
                                   void * userdata2) {
    (void)userdata2;
    DeviceRequestData* data = (DeviceRequestData*)userdata1;
    if (status == WGPURequestDeviceStatus_Success) {
        data->device = device;
    } else {
        fprintf(stderr, "WebGPU device request failed\n");
    }
    data->done = 1;
}



// ====================================================
// Initialization
// ====================================================

int webgpu_init(void) {
    if (g_webgpu_ctx.initialized) {
        return 0; // Already initialized
    }
    
    // Set environment to disable Vulkan fallback
    setenv("DAWN_DEBUG_BREAK_ON_ERROR", "1", 0);
    
    // Create WebGPU instance
    WGPUInstanceDescriptor instance_desc = {0};
    g_webgpu_ctx.instance = wgpuCreateInstance(&instance_desc);
    if (!g_webgpu_ctx.instance) {
        fprintf(stderr, "Failed to create WebGPU instance\n");
        return -1;
    }
    
    // Process any pending events to let Dawn discover backends
    wgpuInstanceProcessEvents(g_webgpu_ctx.instance);
    
    // Try to request adapter with explicit backend type
    // Dawn's WGPUBackendType enum: Metal=2, Vulkan=5
    printf("Attempting to request Metal adapter (backendType=2)...\n");
    
    WGPURequestAdapterOptions adapter_opts = {
        .powerPreference = WGPUPowerPreference_HighPerformance,
        .backendType = WGPUBackendType_Metal,
        .forceFallbackAdapter = 0,
    };
    
    AdapterRequestData adapter_data = {.adapter = NULL, .done = 0};
    WGPURequestAdapterCallbackInfo adapter_callback = {
        .mode = WGPUCallbackMode_AllowSpontaneous,
        .callback = adapter_request_callback,
        .userdata1 = &adapter_data,
        .userdata2 = NULL,
    };
    WGPUFuture adapter_future = wgpuInstanceRequestAdapter(g_webgpu_ctx.instance, &adapter_opts, adapter_callback);
    
    // Poll for adapter request to complete (with timeout)
    int adapter_timeout = 0;
    while (!adapter_data.done && adapter_timeout < 1000) {
        wgpuInstanceProcessEvents(g_webgpu_ctx.instance);
        adapter_timeout++;
        if (adapter_timeout % 100 == 0) {
            printf("Waiting for adapter... (%d)\n", adapter_timeout);
        }
    }
    if (adapter_timeout >= 1000) {
        fprintf(stderr, "Timeout waiting for adapter!\n");
        wgpuInstanceRelease(g_webgpu_ctx.instance);
        g_webgpu_ctx.instance = NULL;
        return -1;
    }
    
    if (!adapter_data.adapter) {
        fprintf(stderr, "Failed to get Metal adapter, trying without backend constraint...\n");
        
        // Try again without specifying backend type
        adapter_opts.backendType = WGPUBackendType_Undefined;
        adapter_data.done = 0;
        adapter_future = wgpuInstanceRequestAdapter(g_webgpu_ctx.instance, &adapter_opts, adapter_callback);
        adapter_timeout = 0;
        while (!adapter_data.done && adapter_timeout < 1000) {
            wgpuInstanceProcessEvents(g_webgpu_ctx.instance);
            adapter_timeout++;
        }
        if (adapter_timeout >= 1000) {
            fprintf(stderr, "Timeout waiting for fallback adapter!\n");
            wgpuInstanceRelease(g_webgpu_ctx.instance);
            g_webgpu_ctx.instance = NULL;
            return -1;
        }
        
        if (!adapter_data.adapter) {
            fprintf(stderr, "Failed to get any WebGPU adapter\n");
            wgpuInstanceRelease(g_webgpu_ctx.instance);
            g_webgpu_ctx.instance = NULL;
            return -1;
        }
    }
    
    // Get adapter properties to verify backend type
    WGPUAdapterInfo adapter_info = {0};
    wgpuAdapterGetInfo(adapter_data.adapter, &adapter_info);
    printf("WebGPU Adapter: %.*s\n", (int)adapter_info.device.length, adapter_info.device.data);
    printf("Backend Type: %d (5=Metal, 6=Vulkan, 4=D3D12, 3=D3D11, 7=OpenGL)\n", adapter_info.backendType);
    printf("Adapter Type: %d (1=Discrete, 2=Integrated, 3=CPU, 4=Unknown)\n", adapter_info.adapterType);
    
    if (adapter_info.backendType == WGPUBackendType_Metal && 
        (adapter_info.adapterType == WGPUAdapterType_DiscreteGPU || 
         adapter_info.adapterType == WGPUAdapterType_IntegratedGPU)) {
        printf("✓ Successfully using Metal GPU backend!\n");
    } else if (adapter_info.adapterType == WGPUAdapterType_CPU) {
        printf("WARNING: Using CPU adapter!\n");
    } else if (adapter_info.backendType != WGPUBackendType_Metal) {
        printf("WARNING: Not using Metal backend!\n");
    }
    
    wgpuAdapterInfoFreeMembers(adapter_info);
    
    g_webgpu_ctx.adapter = adapter_data.adapter;
    
    // Disable validation for production GPU execution
    const char* enabled_toggles[] = {"skip_validation"};
    WGPUDawnTogglesDescriptor toggles_desc = {
        .chain = {
            .sType = WGPUSType_DawnTogglesDescriptor,
        },
        .enabledToggleCount = 1,
        .enabledToggles = enabled_toggles,
    };
    
    // Request device - don't specify limits, let it use defaults
    WGPUDeviceDescriptor device_desc = {
        .nextInChain = (WGPUChainedStruct*)&toggles_desc,
        .label = {.data = "BaseDNN WebGPU Device", .length = WGPU_STRLEN},
        .requiredFeatureCount = 0,
        .requiredLimits = NULL,
        .defaultQueue = {
            .label = {.data = "Default Queue", .length = WGPU_STRLEN},
        },
    };
    
    DeviceRequestData device_data = {.device = NULL, .done = 0};
    WGPURequestDeviceCallbackInfo device_callback = {
        .mode = WGPUCallbackMode_AllowSpontaneous,
        .callback = device_request_callback,
        .userdata1 = &device_data,
        .userdata2 = NULL,
    };
    WGPUFuture device_future = wgpuAdapterRequestDevice(g_webgpu_ctx.adapter, &device_desc, device_callback);
    
    // Poll for device request to complete (with timeout)
    int device_timeout = 0;
    while (!device_data.done && device_timeout < 1000) {
        wgpuInstanceProcessEvents(g_webgpu_ctx.instance);
        device_timeout++;
    }
    if (device_timeout >= 1000) {
        fprintf(stderr, "Timeout waiting for device!\n");
        wgpuAdapterRelease(g_webgpu_ctx.adapter);
        wgpuInstanceRelease(g_webgpu_ctx.instance);
        g_webgpu_ctx.adapter = NULL;
        g_webgpu_ctx.instance = NULL;
        return -1;
    }
    
    if (!device_data.device) {
        fprintf(stderr, "Failed to get WebGPU device\n");
        wgpuAdapterRelease(g_webgpu_ctx.adapter);
        wgpuInstanceRelease(g_webgpu_ctx.instance);
        g_webgpu_ctx.adapter = NULL;
        g_webgpu_ctx.instance = NULL;
        return -1;
    }
    g_webgpu_ctx.device = device_data.device;
    
    // Get queue
    g_webgpu_ctx.queue = wgpuDeviceGetQueue(g_webgpu_ctx.device);
    if (!g_webgpu_ctx.queue) {
        fprintf(stderr, "Failed to get WebGPU queue\n");
        wgpuDeviceRelease(g_webgpu_ctx.device);
        wgpuAdapterRelease(g_webgpu_ctx.adapter);
        wgpuInstanceRelease(g_webgpu_ctx.instance);
        g_webgpu_ctx.device = NULL;
        g_webgpu_ctx.adapter = NULL;
        g_webgpu_ctx.instance = NULL;
        return -1;
    }
    
    g_webgpu_ctx.initialized = 1;
    return 0;
}

void webgpu_cleanup(void) {
    if (g_webgpu_ctx.queue) {
        wgpuQueueRelease(g_webgpu_ctx.queue);
        g_webgpu_ctx.queue = NULL;
    }
    if (g_webgpu_ctx.device) {
        wgpuDeviceRelease(g_webgpu_ctx.device);
        g_webgpu_ctx.device = NULL;
    }
    if (g_webgpu_ctx.adapter) {
        wgpuAdapterRelease(g_webgpu_ctx.adapter);
        g_webgpu_ctx.adapter = NULL;
    }
    if (g_webgpu_ctx.instance) {
        wgpuInstanceRelease(g_webgpu_ctx.instance);
        g_webgpu_ctx.instance = NULL;
    }
    g_webgpu_ctx.initialized = 0;
}

int webgpu_available(void) {
    return g_webgpu_ctx.initialized;
}

// ====================================================
// Context Access
// ====================================================

WGPUDevice webgpu_get_device(void) {
    return g_webgpu_ctx.device;
}

WGPUQueue webgpu_get_queue(void) {
    return g_webgpu_ctx.queue;
}

struct WGPUDeviceImpl* webgpu_get_device_internal(void) {
    return g_webgpu_ctx.device;
}

struct WGPUQueueImpl* webgpu_get_queue_internal(void) {
    return g_webgpu_ctx.queue;
}

// ====================================================
// Buffer Management
// ====================================================

WGPUBuffer webgpu_create_buffer(size_t size, WGPUBufferUsage usage) {
    if (!webgpu_available()) {
        return NULL;
    }
    
    WGPUBufferDescriptor buffer_desc = {
        .label = "Tensor Buffer",
        .size = size,
        .usage = usage,
        .mappedAtCreation = 0,
    };
    
    return wgpuDeviceCreateBuffer(g_webgpu_ctx.device, &buffer_desc);
}

void webgpu_write_buffer(WGPUBuffer buffer, const void* data, size_t size) {
    if (!webgpu_available() || !buffer || !data) {
        return;
    }
    
    wgpuQueueWriteBuffer(g_webgpu_ctx.queue, buffer, 0, data, size);
}

// Callback for buffer mapping
typedef struct {
    void* data;
    size_t size;
    int done;
} BufferMapContext;

static void buffer_map_callback(WGPUMapAsyncStatus status, WGPUStringView message, void* userdata1, void* userdata2) {
    (void)status; (void)message; (void)userdata2;
    BufferMapContext* ctx = (BufferMapContext*)userdata1;
    ctx->done = 1;
}

void webgpu_read_buffer(WGPUBuffer buffer, void* data, size_t size) {
    if (!webgpu_available() || !buffer || !data) {
        return;
    }
    
    BufferMapContext ctx = {.data = data, .size = size, .done = 0};
    
    WGPUBufferMapCallbackInfo callback_info = {
        .mode = WGPUCallbackMode_AllowSpontaneous,
        .callback = buffer_map_callback,
        .userdata1 = &ctx,
        .userdata2 = NULL,
    };
    
    wgpuBufferMapAsync(buffer, WGPUMapMode_Read, 0, size, callback_info);
    
    // Wait for mapping to complete (with timeout)
    int map_timeout = 0;
    while (!ctx.done && map_timeout < 10000) {
        wgpuInstanceProcessEvents(g_webgpu_ctx.instance);
        map_timeout++;
    }
    if (map_timeout >= 10000) {
        fprintf(stderr, "Timeout waiting for buffer map!\n");
        return;
    }
    
    const void* mapped = wgpuBufferGetConstMappedRange(buffer, 0, size);
    if (mapped) {
        memcpy(data, mapped, size);
        wgpuBufferUnmap(buffer);
    }
}

// ====================================================
// Shader Management
// ====================================================

WGPUShaderModule webgpu_create_shader_module(const char* wgsl_code) {
    if (!webgpu_available() || !wgsl_code) {
        return NULL;
    }
    
    WGPUShaderSourceWGSL wgsl_source = {
        .chain = {
            .next = NULL,
            .sType = WGPUSType_ShaderSourceWGSL,
        },
        .code = {.data = wgsl_code, .length = WGPU_STRLEN},
    };
    
    WGPUShaderModuleDescriptor shader_desc = {
        .nextInChain = (WGPUChainedStruct*)&wgsl_source,
        .label = {.data = "Compute Shader", .length = WGPU_STRLEN},
    };
    
    return wgpuDeviceCreateShaderModule(g_webgpu_ctx.device, &shader_desc);
}

// ====================================================
// Backend Registration
// ====================================================

// Forward declarations from webgpu_ops.c
extern Tensor* webgpu_tensor_add(Tensor *A, Tensor *B);
extern Tensor* webgpu_tensor_sub(Tensor *A, Tensor *B);
extern Tensor* webgpu_tensor_mul(Tensor *A, Tensor *B);
extern Tensor* webgpu_tensor_matmul(Tensor *A, Tensor *B);
extern Tensor* webgpu_tensor_relu(Tensor *Z);
extern Tensor* webgpu_tensor_sigmoid(Tensor *Z);
extern Tensor* webgpu_tensor_tanh(Tensor *Z);
extern Tensor* webgpu_tensor_softmax(Tensor *Z);

// Import registry functions
extern void register_operation_backend(const char *name, void *op_fn, int priority);

void webgpu_register_ops(void) {
    if (!webgpu_available()) {
        printf("WARNING: webgpu_register_ops called but WebGPU not available!\n");
        return;
    }
    
    printf("Registering WebGPU operations with priority 10...\n");
    // Register WebGPU-accelerated operations with priority 10 (higher than CPU's 0)
    register_operation_backend("add", (void*)webgpu_tensor_add, 10);
    register_operation_backend("sub", (void*)webgpu_tensor_sub, 10);
    register_operation_backend("mul", (void*)webgpu_tensor_mul, 10);
    register_operation_backend("matmul", (void*)webgpu_tensor_matmul, 10);
    register_operation_backend("relu", (void*)webgpu_tensor_relu, 10);
    register_operation_backend("sigmoid", (void*)webgpu_tensor_sigmoid, 10);
    register_operation_backend("tanh", (void*)webgpu_tensor_tanh, 10);
    register_operation_backend("softmax", (void*)webgpu_tensor_softmax, 10);
    printf("WebGPU operations registered successfully!\n");
}

// ====================================================
// Backend Initialization (called from registry_init)
// ====================================================

void backend_init_all(void) {
#ifdef HAS_WEBGPU
    if (webgpu_init() == 0) {
        webgpu_register_ops();
    }
#endif
}
