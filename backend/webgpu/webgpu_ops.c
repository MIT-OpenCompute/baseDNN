#include "webgpu_backend.h"
#include "../../core/include/ops.h"
#include <webgpu/webgpu.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// External WebGPU context access
extern int webgpu_available(void);
extern WGPUDevice webgpu_get_device(void);
extern WGPUQueue webgpu_get_queue(void);
extern WGPUBuffer webgpu_create_buffer(size_t size, WGPUBufferUsage usage);
extern void webgpu_write_buffer(WGPUBuffer buffer, const void* data, size_t size);
extern void webgpu_read_buffer(WGPUBuffer buffer, void* data, size_t size);
extern WGPUShaderModule webgpu_create_shader_module(const char* wgsl_code);

// CPU fallback functions (from ops.c)
extern Tensor* tensor_add_cpu(Tensor *A, Tensor *B);
extern Tensor* tensor_sub_cpu(Tensor *A, Tensor *B);
extern Tensor* tensor_mul_cpu(Tensor *A, Tensor *B);
extern Tensor* tensor_matmul_cpu(Tensor *A, Tensor *B);
extern Tensor* tensor_relu_cpu(Tensor *Z);
extern Tensor* tensor_sigmoid_cpu(Tensor *Z);
extern Tensor* tensor_tanh_cpu(Tensor *Z);
extern Tensor* tensor_softmax_cpu(Tensor *Z);

// ====================================================
// Shader Sources (embedded WGSL)
// ====================================================

#include "shaders/shader_sources.h"

// ====================================================
// Helper: Setup Autograd for WebGPU Operations
// ====================================================

static void setup_autograd_two_inputs(Tensor *A, Tensor *B, Tensor *C, const char *op_name, void (*backward_fn)(Tensor *)) {
    if (A->requires_grad || B->requires_grad) {
        C->requires_grad = 1;
        C->op_name = strdup(op_name);
        C->num_inputs = 2;
        C->inputs = (Tensor **)malloc(2 * sizeof(Tensor *));
        C->inputs[0] = A;
        C->inputs[1] = B;
        C->backward_fn = backward_fn;
    }
}

static void setup_autograd_one_input(Tensor *Z, Tensor *A, const char *op_name, void (*backward_fn)(Tensor *)) {
    if (Z->requires_grad) {
        A->requires_grad = 1;
        A->op_name = strdup(op_name);
        A->num_inputs = 1;
        A->inputs = (Tensor **)malloc(sizeof(Tensor *));
        A->inputs[0] = Z;
        A->backward_fn = backward_fn;
    }
}

// ====================================================
// Pipeline Cache (for performance)
// ====================================================

typedef struct {
    const char* name;
    WGPUComputePipeline pipeline;
} PipelineCache;

#define MAX_PIPELINES 16
static PipelineCache g_pipeline_cache[MAX_PIPELINES] = {0};
static int g_pipeline_count = 0;

static WGPUComputePipeline get_or_create_pipeline(const char* name, const char* wgsl_code, 
                                                  WGPUBindGroupLayout bind_group_layout) {
    // Check cache
    for (int i = 0; i < g_pipeline_count; i++) {
        if (strcmp(g_pipeline_cache[i].name, name) == 0) {
            return g_pipeline_cache[i].pipeline;
        }
    }
    
    // Create new pipeline
    WGPUShaderModule shader = webgpu_create_shader_module(wgsl_code);
    if (!shader) return NULL;
    
    WGPUPipelineLayoutDescriptor layout_desc = {
        .bindGroupLayoutCount = 1,
        .bindGroupLayouts = &bind_group_layout,
    };
    WGPUPipelineLayout pipeline_layout = wgpuDeviceCreatePipelineLayout(webgpu_get_device(), &layout_desc);
    
    WGPUComputePipelineDescriptor pipeline_desc = {
        .layout = pipeline_layout,
        .compute = {
            .module = shader,
            .entryPoint = {.data = "main", .length = WGPU_STRLEN},
        },
    };
    
    WGPUComputePipeline pipeline = wgpuDeviceCreateComputePipeline(webgpu_get_device(), &pipeline_desc);
    
    wgpuPipelineLayoutRelease(pipeline_layout);
    wgpuShaderModuleRelease(shader);
    
    // Cache it
    if (g_pipeline_count < MAX_PIPELINES) {
        g_pipeline_cache[g_pipeline_count].name = name;
        g_pipeline_cache[g_pipeline_count].pipeline = pipeline;
        g_pipeline_count++;
    }
    
    return pipeline;
}

// ====================================================
// Element-wise Binary Operations
// ====================================================

static Tensor* elementwise_binary_op(Tensor *A, Tensor *B, const char* op_name, 
                                    const char* shader_code, void (*backward_fn)(Tensor *),
                                    Tensor* (*cpu_fallback)(Tensor*, Tensor*)) {
    static int dispatch_count = 0;
    if (!webgpu_available() || !A || !B) {
        return cpu_fallback(A, B);
    }
    
    if (dispatch_count < 5) {
        printf("[GPU] Dispatching %s compute shader (size=%zu)\n", op_name, A->size);
        dispatch_count++;
    }
    
    // Check shape compatibility
    if (A->ndim != B->ndim) {
        return cpu_fallback(A, B);
    }
    for (size_t i = 0; i < A->ndim; i++) {
        if (A->shape[i] != B->shape[i]) {
            return cpu_fallback(A, B);
        }
    }
    
    // Create output tensor
    Tensor *C = tensor_create(A->shape, A->ndim);
    if (!C) return NULL;
    
    size_t buffer_size = A->size * sizeof(float);
    
    // Create GPU buffers
    WGPUBuffer buffer_a = webgpu_create_buffer(buffer_size, 
        WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst);
    WGPUBuffer buffer_b = webgpu_create_buffer(buffer_size,
        WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst);
    WGPUBuffer buffer_c = webgpu_create_buffer(buffer_size,
        WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc);
    
    // Upload data
    webgpu_write_buffer(buffer_a, A->data, buffer_size);
    webgpu_write_buffer(buffer_b, B->data, buffer_size);
    
    // Create bind group layout
    WGPUBindGroupLayoutEntry layout_entries[3] = {
        {
            .binding = 0,
            .visibility = WGPUShaderStage_Compute,
            .buffer = {.type = WGPUBufferBindingType_ReadOnlyStorage},
        },
        {
            .binding = 1,
            .visibility = WGPUShaderStage_Compute,
            .buffer = {.type = WGPUBufferBindingType_ReadOnlyStorage},
        },
        {
            .binding = 2,
            .visibility = WGPUShaderStage_Compute,
            .buffer = {.type = WGPUBufferBindingType_Storage},
        },
    };
    
    WGPUBindGroupLayoutDescriptor layout_desc = {
        .entryCount = 3,
        .entries = layout_entries,
    };
    WGPUBindGroupLayout bind_group_layout = wgpuDeviceCreateBindGroupLayout(webgpu_get_device(), &layout_desc);
    
    // Get or create pipeline
    WGPUComputePipeline pipeline = get_or_create_pipeline(op_name, shader_code, bind_group_layout);
    
    // Create bind group
    WGPUBindGroupEntry bind_entries[3] = {
        {.binding = 0, .buffer = buffer_a, .size = buffer_size},
        {.binding = 1, .buffer = buffer_b, .size = buffer_size},
        {.binding = 2, .buffer = buffer_c, .size = buffer_size},
    };
    
    WGPUBindGroupDescriptor bind_group_desc = {
        .layout = bind_group_layout,
        .entryCount = 3,
        .entries = bind_entries,
    };
    WGPUBindGroup bind_group = wgpuDeviceCreateBindGroup(webgpu_get_device(), &bind_group_desc);
    
    // Encode and submit
    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(webgpu_get_device(), NULL);
    WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(encoder, NULL);
    
    wgpuComputePassEncoderSetPipeline(pass, pipeline);
    wgpuComputePassEncoderSetBindGroup(pass, 0, bind_group, 0, NULL);
    
    uint32_t workgroups = (A->size + 255) / 256;
    wgpuComputePassEncoderDispatchWorkgroups(pass, workgroups, 1, 1);
    wgpuComputePassEncoderEnd(pass);
    
    WGPUCommandBuffer commands = wgpuCommandEncoderFinish(encoder, NULL);
    wgpuQueueSubmit(webgpu_get_queue(), 1, &commands);
    
    // Read result back
    WGPUBuffer staging_buffer = webgpu_create_buffer(buffer_size,
        WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead);
    
    WGPUCommandEncoder copy_encoder = wgpuDeviceCreateCommandEncoder(webgpu_get_device(), NULL);
    wgpuCommandEncoderCopyBufferToBuffer(copy_encoder, buffer_c, 0, staging_buffer, 0, buffer_size);
    WGPUCommandBuffer copy_commands = wgpuCommandEncoderFinish(copy_encoder, NULL);
    wgpuQueueSubmit(webgpu_get_queue(), 1, &copy_commands);
    
    webgpu_read_buffer(staging_buffer, C->data, buffer_size);
    
    // Cleanup
    wgpuCommandBufferRelease(copy_commands);
    wgpuCommandEncoderRelease(copy_encoder);
    wgpuBufferRelease(staging_buffer);
    wgpuCommandBufferRelease(commands);
    wgpuComputePassEncoderRelease(pass);
    wgpuCommandEncoderRelease(encoder);
    wgpuBindGroupRelease(bind_group);
    wgpuBindGroupLayoutRelease(bind_group_layout);
    wgpuBufferRelease(buffer_a);
    wgpuBufferRelease(buffer_b);
    wgpuBufferRelease(buffer_c);
    
    // Setup autograd
    setup_autograd_two_inputs(A, B, C, op_name, backward_fn);
    
    return C;
}

// ====================================================
// Element-wise Operations
// ====================================================

Tensor* webgpu_tensor_add(Tensor *A, Tensor *B) {
    static int call_count = 0;
    if (call_count < 3) {
        printf("[DEBUG] webgpu_tensor_add called! (webgpu_available=%d)\n", webgpu_available());
        call_count++;
    }
    return elementwise_binary_op(A, B, "add", SHADER_ADD, backward_add, tensor_add_cpu);
}

Tensor* webgpu_tensor_sub(Tensor *A, Tensor *B) {
    return elementwise_binary_op(A, B, "sub", SHADER_SUB, backward_sub, tensor_sub_cpu);
}

Tensor* webgpu_tensor_mul(Tensor *A, Tensor *B) {
    return elementwise_binary_op(A, B, "mul", SHADER_MUL, backward_mul, tensor_mul_cpu);
}

// ====================================================
// Matrix Multiplication
// ====================================================

Tensor* webgpu_tensor_matmul(Tensor *A, Tensor *B) {
    if (!webgpu_available() || !A || !B) {
        return tensor_matmul_cpu(A, B);
    }
    
    if (A->ndim != 2 || B->ndim != 2 || A->shape[1] != B->shape[0]) {
        return tensor_matmul_cpu(A, B);
    }
    
    uint32_t M = A->shape[0];
    uint32_t K = A->shape[1];
    uint32_t N = B->shape[1];
    
    size_t output_shape[2] = {M, N};
    Tensor *C = tensor_create(output_shape, 2);
    if (!C) return NULL;
    
    // Create buffers
    WGPUBuffer buffer_a = webgpu_create_buffer(A->size * sizeof(float),
        WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst);
    WGPUBuffer buffer_b = webgpu_create_buffer(B->size * sizeof(float),
        WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst);
    WGPUBuffer buffer_c = webgpu_create_buffer(C->size * sizeof(float),
        WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc);
    
    // Uniform buffer for dimensions
    struct {uint32_t M, K, N, pad;} dims = {M, K, N, 0};
    WGPUBuffer uniform_buffer = webgpu_create_buffer(sizeof(dims),
        WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst);
    
    webgpu_write_buffer(buffer_a, A->data, A->size * sizeof(float));
    webgpu_write_buffer(buffer_b, B->data, B->size * sizeof(float));
    webgpu_write_buffer(uniform_buffer, &dims, sizeof(dims));
    
    // Create bind group layout
    WGPUBindGroupLayoutEntry layout_entries[4] = {
        {.binding = 0, .visibility = WGPUShaderStage_Compute,
         .buffer = {.type = WGPUBufferBindingType_ReadOnlyStorage}},
        {.binding = 1, .visibility = WGPUShaderStage_Compute,
         .buffer = {.type = WGPUBufferBindingType_ReadOnlyStorage}},
        {.binding = 2, .visibility = WGPUShaderStage_Compute,
         .buffer = {.type = WGPUBufferBindingType_Storage}},
        {.binding = 3, .visibility = WGPUShaderStage_Compute,
         .buffer = {.type = WGPUBufferBindingType_Uniform}},
    };
    
    WGPUBindGroupLayoutDescriptor layout_desc = {.entryCount = 4, .entries = layout_entries};
    WGPUBindGroupLayout bind_group_layout = wgpuDeviceCreateBindGroupLayout(webgpu_get_device(), &layout_desc);
    
    WGPUComputePipeline pipeline = get_or_create_pipeline("matmul", SHADER_MATMUL, bind_group_layout);
    
    // Create bind group
    WGPUBindGroupEntry bind_entries[4] = {
        {.binding = 0, .buffer = buffer_a, .size = A->size * sizeof(float)},
        {.binding = 1, .buffer = buffer_b, .size = B->size * sizeof(float)},
        {.binding = 2, .buffer = buffer_c, .size = C->size * sizeof(float)},
        {.binding = 3, .buffer = uniform_buffer, .size = sizeof(dims)},
    };
    
    WGPUBindGroupDescriptor bind_group_desc = {.layout = bind_group_layout, .entryCount = 4, .entries = bind_entries};
    WGPUBindGroup bind_group = wgpuDeviceCreateBindGroup(webgpu_get_device(), &bind_group_desc);
    
    // Dispatch
    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(webgpu_get_device(), NULL);
    WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(encoder, NULL);
    
    wgpuComputePassEncoderSetPipeline(pass, pipeline);
    wgpuComputePassEncoderSetBindGroup(pass, 0, bind_group, 0, NULL);
    
    uint32_t workgroups_x = (N + 15) / 16;
    uint32_t workgroups_y = (M + 15) / 16;
    wgpuComputePassEncoderDispatchWorkgroups(pass, workgroups_x, workgroups_y, 1);
    wgpuComputePassEncoderEnd(pass);
    
    WGPUCommandBuffer commands = wgpuCommandEncoderFinish(encoder, NULL);
    wgpuQueueSubmit(webgpu_get_queue(), 1, &commands);
    
    // Read back
    WGPUBuffer staging = webgpu_create_buffer(C->size * sizeof(float),
        WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead);
    
    WGPUCommandEncoder copy_enc = wgpuDeviceCreateCommandEncoder(webgpu_get_device(), NULL);
    wgpuCommandEncoderCopyBufferToBuffer(copy_enc, buffer_c, 0, staging, 0, C->size * sizeof(float));
    WGPUCommandBuffer copy_cmd = wgpuCommandEncoderFinish(copy_enc, NULL);
    wgpuQueueSubmit(webgpu_get_queue(), 1, &copy_cmd);
    
    webgpu_read_buffer(staging, C->data, C->size * sizeof(float));
    
    // Cleanup
    wgpuCommandBufferRelease(copy_cmd);
    wgpuCommandEncoderRelease(copy_enc);
    wgpuBufferRelease(staging);
    wgpuCommandBufferRelease(commands);
    wgpuComputePassEncoderRelease(pass);
    wgpuCommandEncoderRelease(encoder);
    wgpuBindGroupRelease(bind_group);
    wgpuBindGroupLayoutRelease(bind_group_layout);
    wgpuBufferRelease(buffer_a);
    wgpuBufferRelease(buffer_b);
    wgpuBufferRelease(buffer_c);
    wgpuBufferRelease(uniform_buffer);
    
    setup_autograd_two_inputs(A, B, C, "matmul", backward_matmul);
    
    return C;
}

// ====================================================
// Element-wise Unary Operations (Activations)
// ====================================================

static Tensor* elementwise_unary_op(Tensor *Z, const char* op_name, const char* shader_code,
                                   void (*backward_fn)(Tensor *), Tensor* (*cpu_fallback)(Tensor*)) {
    if (!webgpu_available() || !Z) {
        return cpu_fallback(Z);
    }
    
    Tensor *A = tensor_create(Z->shape, Z->ndim);
    if (!A) return NULL;
    
    size_t buffer_size = Z->size * sizeof(float);
    
    WGPUBuffer buffer_in = webgpu_create_buffer(buffer_size,
        WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst);
    WGPUBuffer buffer_out = webgpu_create_buffer(buffer_size,
        WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc);
    
    webgpu_write_buffer(buffer_in, Z->data, buffer_size);
    
    // Bind group layout
    WGPUBindGroupLayoutEntry layout_entries[2] = {
        {.binding = 0, .visibility = WGPUShaderStage_Compute,
         .buffer = {.type = WGPUBufferBindingType_ReadOnlyStorage}},
        {.binding = 1, .visibility = WGPUShaderStage_Compute,
         .buffer = {.type = WGPUBufferBindingType_Storage}},
    };
    
    WGPUBindGroupLayoutDescriptor layout_desc = {.entryCount = 2, .entries = layout_entries};
    WGPUBindGroupLayout bind_group_layout = wgpuDeviceCreateBindGroupLayout(webgpu_get_device(), &layout_desc);
    
    WGPUComputePipeline pipeline = get_or_create_pipeline(op_name, shader_code, bind_group_layout);
    
    WGPUBindGroupEntry bind_entries[2] = {
        {.binding = 0, .buffer = buffer_in, .size = buffer_size},
        {.binding = 1, .buffer = buffer_out, .size = buffer_size},
    };
    
    WGPUBindGroupDescriptor bind_group_desc = {.layout = bind_group_layout, .entryCount = 2, .entries = bind_entries};
    WGPUBindGroup bind_group = wgpuDeviceCreateBindGroup(webgpu_get_device(), &bind_group_desc);
    
    // Dispatch
    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(webgpu_get_device(), NULL);
    WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(encoder, NULL);
    
    wgpuComputePassEncoderSetPipeline(pass, pipeline);
    wgpuComputePassEncoderSetBindGroup(pass, 0, bind_group, 0, NULL);
    
    uint32_t workgroups = (Z->size + 255) / 256;
    wgpuComputePassEncoderDispatchWorkgroups(pass, workgroups, 1, 1);
    wgpuComputePassEncoderEnd(pass);
    
    WGPUCommandBuffer commands = wgpuCommandEncoderFinish(encoder, NULL);
    wgpuQueueSubmit(webgpu_get_queue(), 1, &commands);
    
    // Read back
    WGPUBuffer staging = webgpu_create_buffer(buffer_size,
        WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead);
    
    WGPUCommandEncoder copy_enc = wgpuDeviceCreateCommandEncoder(webgpu_get_device(), NULL);
    wgpuCommandEncoderCopyBufferToBuffer(copy_enc, buffer_out, 0, staging, 0, buffer_size);
    WGPUCommandBuffer copy_cmd = wgpuCommandEncoderFinish(copy_enc, NULL);
    wgpuQueueSubmit(webgpu_get_queue(), 1, &copy_cmd);
    
    webgpu_read_buffer(staging, A->data, buffer_size);
    
    // Cleanup
    wgpuCommandBufferRelease(copy_cmd);
    wgpuCommandEncoderRelease(copy_enc);
    wgpuBufferRelease(staging);
    wgpuCommandBufferRelease(commands);
    wgpuComputePassEncoderRelease(pass);
    wgpuCommandEncoderRelease(encoder);
    wgpuBindGroupRelease(bind_group);
    wgpuBindGroupLayoutRelease(bind_group_layout);
    wgpuBufferRelease(buffer_in);
    wgpuBufferRelease(buffer_out);
    
    setup_autograd_one_input(Z, A, op_name, backward_fn);
    
    return A;
}

Tensor* webgpu_tensor_relu(Tensor *Z) {
    return elementwise_unary_op(Z, "relu", SHADER_RELU, backward_relu, tensor_relu);
}

Tensor* webgpu_tensor_sigmoid(Tensor *Z) {
    return elementwise_unary_op(Z, "sigmoid", SHADER_SIGMOID, backward_sigmoid, tensor_sigmoid);
}

Tensor* webgpu_tensor_tanh(Tensor *Z) {
    return elementwise_unary_op(Z, "tanh", SHADER_TANH, backward_tanh, tensor_tanh);
}

// ====================================================
// Softmax (Special Case - needs batch handling)
// ====================================================

Tensor* webgpu_tensor_softmax(Tensor *Z) {
    if (!webgpu_available() || !Z) {
        return tensor_softmax_cpu(Z);
    }
    
    // Simplified: treat as 2D [batch, features]
    if (Z->ndim < 2) {
        return tensor_softmax_cpu(Z); // Fallback for 1D
    }
    
    Tensor *A = tensor_create(Z->shape, Z->ndim);
    if (!A) return NULL;
    
    uint32_t batch = Z->shape[0];
    uint32_t size = Z->shape[Z->ndim - 1];
    uint32_t stride = size;
    
    // For multi-dimensional, compute stride
    for (size_t i = 1; i < Z->ndim; i++) {
        stride *= Z->shape[i];
    }
    stride /= batch;
    
    size_t buffer_size = Z->size * sizeof(float);
    
    WGPUBuffer buffer_in = webgpu_create_buffer(buffer_size,
        WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst);
    WGPUBuffer buffer_out = webgpu_create_buffer(buffer_size,
        WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc);
    
    struct {uint32_t size; uint32_t stride; uint32_t pad1; uint32_t pad2;} params = {size, stride, 0, 0};
    WGPUBuffer uniform = webgpu_create_buffer(sizeof(params),
        WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst);
    
    webgpu_write_buffer(buffer_in, Z->data, buffer_size);
    webgpu_write_buffer(uniform, &params, sizeof(params));
    
    // Bind group layout
    WGPUBindGroupLayoutEntry layout_entries[3] = {
        {.binding = 0, .visibility = WGPUShaderStage_Compute,
         .buffer = {.type = WGPUBufferBindingType_ReadOnlyStorage}},
        {.binding = 1, .visibility = WGPUShaderStage_Compute,
         .buffer = {.type = WGPUBufferBindingType_Storage}},
        {.binding = 2, .visibility = WGPUShaderStage_Compute,
         .buffer = {.type = WGPUBufferBindingType_Uniform}},
    };
    
    WGPUBindGroupLayoutDescriptor layout_desc = {.entryCount = 3, .entries = layout_entries};
    WGPUBindGroupLayout bind_group_layout = wgpuDeviceCreateBindGroupLayout(webgpu_get_device(), &layout_desc);
    
    WGPUComputePipeline pipeline = get_or_create_pipeline("softmax", SHADER_SOFTMAX, bind_group_layout);
    
    WGPUBindGroupEntry bind_entries[3] = {
        {.binding = 0, .buffer = buffer_in, .size = buffer_size},
        {.binding = 1, .buffer = buffer_out, .size = buffer_size},
        {.binding = 2, .buffer = uniform, .size = sizeof(params)},
    };
    
    WGPUBindGroupDescriptor bind_group_desc = {.layout = bind_group_layout, .entryCount = 3, .entries = bind_entries};
    WGPUBindGroup bind_group = wgpuDeviceCreateBindGroup(webgpu_get_device(), &bind_group_desc);
    
    // Dispatch (one workgroup per batch element)
    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(webgpu_get_device(), NULL);
    WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(encoder, NULL);
    
    wgpuComputePassEncoderSetPipeline(pass, pipeline);
    wgpuComputePassEncoderSetBindGroup(pass, 0, bind_group, 0, NULL);
    wgpuComputePassEncoderDispatchWorkgroups(pass, batch, 1, 1);
    wgpuComputePassEncoderEnd(pass);
    
    WGPUCommandBuffer commands = wgpuCommandEncoderFinish(encoder, NULL);
    wgpuQueueSubmit(webgpu_get_queue(), 1, &commands);
    
    // Read back
    WGPUBuffer staging = webgpu_create_buffer(buffer_size,
        WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead);
    
    WGPUCommandEncoder copy_enc = wgpuDeviceCreateCommandEncoder(webgpu_get_device(), NULL);
    wgpuCommandEncoderCopyBufferToBuffer(copy_enc, buffer_out, 0, staging, 0, buffer_size);
    WGPUCommandBuffer copy_cmd = wgpuCommandEncoderFinish(copy_enc, NULL);
    wgpuQueueSubmit(webgpu_get_queue(), 1, &copy_cmd);
    
    webgpu_read_buffer(staging, A->data, buffer_size);
    
    // Cleanup
    wgpuCommandBufferRelease(copy_cmd);
    wgpuCommandEncoderRelease(copy_enc);
    wgpuBufferRelease(staging);
    wgpuCommandBufferRelease(commands);
    wgpuComputePassEncoderRelease(pass);
    wgpuCommandEncoderRelease(encoder);
    wgpuBindGroupRelease(bind_group);
    wgpuBindGroupLayoutRelease(bind_group_layout);
    wgpuBufferRelease(buffer_in);
    wgpuBufferRelease(buffer_out);
    wgpuBufferRelease(uniform);
    
    setup_autograd_one_input(Z, A, "softmax", backward_softmax);
    
    return A;
}
