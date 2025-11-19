// Softmax activation: output = exp(input - max) / sum(exp(input - max))
// Two-pass algorithm: 1) find max and sum, 2) normalize

struct SoftmaxParams {
    size: u32,
    stride: u32,
    pad1: u32,
    pad2: u32,
};

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: SoftmaxParams;

// Workgroup shared memory for reduction
var<workgroup> shared_max: array<f32, 256>;
var<workgroup> shared_sum: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let batch_idx = workgroup_id.x;
    let local_idx = local_id.x;
    let offset = batch_idx * params.stride;
    
    // Phase 1: Find max value (for numerical stability)
    var local_max: f32 = -3.402823466e+38; // -FLT_MAX
    for (var i = local_idx; i < params.size; i = i + 256u) {
        local_max = max(local_max, input[offset + i]);
    }
    shared_max[local_idx] = local_max;
    workgroupBarrier();
    
    // Reduce max across workgroup
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (local_idx < s) {
            shared_max[local_idx] = max(shared_max[local_idx], shared_max[local_idx + s]);
        }
        workgroupBarrier();
    }
    let max_val = shared_max[0];
    
    // Phase 2: Compute exp(x - max) and sum
    var local_sum: f32 = 0.0;
    for (var i = local_idx; i < params.size; i = i + 256u) {
        let exp_val = exp(input[offset + i] - max_val);
        output[offset + i] = exp_val;
        local_sum = local_sum + exp_val;
    }
    shared_sum[local_idx] = local_sum;
    workgroupBarrier();
    
    // Reduce sum across workgroup
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (local_idx < s) {
            shared_sum[local_idx] = shared_sum[local_idx] + shared_sum[local_idx + s];
        }
        workgroupBarrier();
    }
    let sum_val = shared_sum[0];
    
    // Phase 3: Normalize
    for (var i = local_idx; i < params.size; i = i + 256u) {
        output[offset + i] = output[offset + i] / sum_val;
    }
}
