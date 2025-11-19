# Plugins Pattern

## Core

### Adding a New Tensor Operation (with Autograd)

All tensor operations are now fully extensible through the registry system. Operations are identified by string names rather than hardcoded enums.

```c
// 1. Implement forward function
Tensor* tensor_my_operation(Tensor *a, Tensor *b) {
    if (!a || !b) return NULL;
    
    // Compute forward pass
    Tensor *output = tensor_create(output_shape, ndim);
    
    // Perform computation
    for (size_t i = 0; i < output->size; i++) {
        output->data[i] = /* your computation */;
    }
    
    // Set up autograd if inputs require gradients
    if (a->requires_grad || b->requires_grad) {
        output->requires_grad = 1;
        output->op_name = strdup("my_operation");  // String-based operation name
        output->num_inputs = 2;
        output->inputs = (Tensor **)malloc(2 * sizeof(Tensor *));
        output->inputs[0] = a;
        output->inputs[1] = b;
        output->backward_fn = get_tensor_op_backward_fn("my_operation");
        
        // Optional: store extra data for backward pass
        // output->extra_data = malloc(sizeof(MyData));
    }
    
    return output;
}

// 2. Implement backward function
void backward_my_operation(Tensor *output) {
    if (!output || !output->inputs) return;
    
    Tensor *a = output->inputs[0];
    Tensor *b = output->inputs[1];
    
    // Compute gradients for inputs
    if (a->requires_grad) {
        if (!a->grad) a->grad = (float *)calloc(a->size, sizeof(float));
        for (size_t i = 0; i < a->size; i++) {
            a->grad[i] += /* gradient computation */ * output->grad[i];
        }
    }
    
    if (b->requires_grad) {
        if (!b->grad) b->grad = (float *)calloc(b->size, sizeof(float));
        for (size_t i = 0; i < b->size; i++) {
            b->grad[i] += /* gradient computation */ * output->grad[i];
        }
    }
}

// 3. Register the backward function in registry_init() or at runtime
register_tensor_op("my_operation", backward_my_operation);

// 4. For loss functions, also register as an operation (for network_train)
register_operation("my_operation", tensor_my_operation);
// Or use the convenience alias:
register_loss("my_loss", tensor_my_loss);
```

### Adding a New Layer

```c
// 1. Define parameters struct (if needed)
typedef struct MyLayerParams {
    size_t param1;
    float param2;
} MyLayerParams;

// 2. Define macro for easy creation
#define MYLAYER(p1, p2) (LayerConfig){ .name = "mylayer", .params = &(MyLayerParams){ p1, p2 } }

// 3. Implement creator function
static Layer* mylayer_create(LayerConfig *config) {
    MyLayerParams *p = (MyLayerParams*)config->params;
    Layer *layer = malloc(sizeof(Layer));
    layer->name = strdup(config->name);
    
    // Allocate parameters (weights, biases, etc.)
    // Store serialization data
    layer->config_data = malloc(sizeof(MyLayerParams));
    memcpy(layer->config_data, p, sizeof(MyLayerParams));
    layer->config_data_size = sizeof(MyLayerParams);
    
    return layer;
}

// 4. Implement forward function
static Tensor* mylayer_forward(Layer *self, Tensor *input) {
    // Your forward pass logic
    return output;
}

// 5. Register in basednn_init()
register_layer("mylayer", mylayer_create, mylayer_forward);
```

### Adding a New Optimizer

```c
// 1. Define parameters struct
typedef struct MyOptimizerParams {
    float learning_rate;
    float beta;
} MyOptimizerParams;

// 2. Define state struct
typedef struct MyOptimizerState {
    float learning_rate;
    float beta;
    Tensor **momentum;
} MyOptimizerState;

// 3. Define macro
#define MYOPT(lr, b) (OptimizerConfig){ .name = "myopt", .params = &(MyOptimizerParams){ lr, b } }

// 4. Implement state initialization
static void* myopt_init_state(Tensor **parameters, size_t num_parameters, void *params) {
    MyOptimizerParams *p = (MyOptimizerParams*)params;
    MyOptimizerState *state = malloc(sizeof(MyOptimizerState));
    state->learning_rate = p->learning_rate;
    state->beta = p->beta;
    
    // Initialize optimizer state tensors
    state->momentum = malloc(num_parameters * sizeof(Tensor*));
    for (size_t i = 0; i < num_parameters; i++) {
        state->momentum[i] = tensor_create(parameters[i]->shape, parameters[i]->ndim);
        tensor_fill(state->momentum[i], 0.0f);
    }
    return state;
}

// 5. Implement step function
static void myopt_step(Optimizer *opt) {
    MyOptimizerState *state = (MyOptimizerState*)opt->state;
    // Update parameters using gradients and state
}

// 6. Implement cleanup
static void myopt_free_state(void *state, size_t num_parameters) {
    MyOptimizerState *s = (MyOptimizerState*)state;
    for (size_t i = 0; i < num_parameters; i++) {
        tensor_free(s->momentum[i]);
    }
    free(s->momentum);
    free(s);
}

// 7. Register in basednn_init()
register_optimizer("myopt", myopt_init_state, myopt_step, myopt_free_state);
```

### Usage

```c
// Initialize registry (registers all built-in operations)
registry_init();

// Register custom operations
register_tensor_op("my_operation", backward_my_operation);
register_operation("my_operation", tensor_my_operation);

// Create network with custom components
Network *net = network_create();
network_add_layer(net, layer_create(MYLAYER(128, 0.5f)));
network_add_layer(net, layer_create(LINEAR(128, 10)));

// Create optimizer
Optimizer *opt = optimizer_create(net->parameters, net->num_parameters, MYOPT(0.01f, 0.9f));

// Train with custom loss
network_train(net, opt, inputs, targets, 10, 32, "my_loss", 1);

// Cleanup
registry_cleanup();
```

### Registry System

The registry system provides complete extensibility for:

1. **Tensor Operations**: Forward and backward functions for autograd
   - `register_tensor_op(name, backward_fn)` - Register backward function
   - `get_tensor_op_backward_fn(name)` - Retrieve backward function
   
2. **Operations/Losses**: High-level operation functions
   - `register_operation(name, op_fn)` - Register operation
   - `register_loss(name, loss_fn)` - Convenience alias for losses
   - `get_operation_fn(name)` - Retrieve operation function

3. **Layers**: Layer creation and forward pass
   - `register_layer(name, create_fn, forward_fn)` - Register layer type
   - `get_layer_create_fn(name)` - Retrieve layer creator
   - `get_layer_forward_fn(name)` - Retrieve layer forward function

4. **Optimizers**: Optimizer initialization, stepping, and cleanup
   - `register_optimizer(name, init_fn, step_fn, free_fn)` - Register optimizer
   - `get_optimizer_init_state_fn(name)` - Retrieve state initializer
   - `get_optimizer_step_fn(name)` - Retrieve step function
   - `get_optimizer_free_state_fn(name)` - Retrieve cleanup function

All built-in operations, layers, and optimizers are automatically registered when you call `registry_init()`.

## Backend

### Adding New Operations

To add a WebGPU-accelerated operation:

1. **Create WGSL shader** in `backend/webgpu/shaders/my_op.wgsl`:
```wgsl
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < arrayLength(&output)) {
        output[idx] = /* your computation */;
    }
}
```

2. **Add shader to** `backend/webgpu/shaders/shader_sources.h`:
```c
static const char* SHADER_MY_OP = 
"@group(0) @binding(0) var<storage, read> input: array<f32>;\n"
"@group(0) @binding(1) var<storage, read_write> output: array<f32>;\n"
"\n"
"@compute @workgroup_size(256)\n"
"fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {\n"
"    let idx = global_id.x;\n"
"    if (idx < arrayLength(&output)) {\n"
"        output[idx] = /* your computation */;\n"
"    }\n"
"}\n";
```

3. **Implement in `webgpu_ops.c`**:
```c
Tensor* webgpu_tensor_my_op(Tensor *A) {
    if (!webgpu_available() || !A) {
        return tensor_my_op(A); // CPU fallback
    }
    
    // Use elementwise_unary_op helper for simple operations
    return elementwise_unary_op(A, "my_op", SHADER_MY_OP, 
                               backward_my_op, tensor_my_op);
}
```

4. **Register in `webgpu_backend.c`**:
```c
void webgpu_register_ops(void) {
    // ... existing registrations
    register_operation_backend("my_op", (void*)webgpu_tensor_my_op, 10);
}
```

5. **Add forward declaration** in `webgpu_backend.c`:
```c
extern Tensor* webgpu_tensor_my_op(Tensor *A);
```

### WebGPU Setup

To build with WebGPU support:

1. **Download wgpu-native** from https://github.com/gfx-rs/wgpu-native/releases
2. **Set environment variable**:
   ```bash
   export WEBGPU_DIR=/path/to/wgpu-native
   ```
3. **Build with CMake**:
   ```bash
   mkdir build && cd build
   cmake .. -DENABLE_WEBGPU=ON
   make
   ```

The WebGPU backend provides cross-platform GPU acceleration on macOS, Linux, Windows, and Web (via Emscripten).
