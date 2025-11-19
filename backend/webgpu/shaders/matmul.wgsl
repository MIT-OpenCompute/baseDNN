// Matrix multiplication: C = A @ B
// A: [M, K], B: [K, N], C: [M, N]
// Tiled algorithm with workgroup memory

struct Dimensions {
    M: u32,
    K: u32,
    N: u32,
    pad: u32,
};

@group(0) @binding(0) var<storage, read> matrix_a: array<f32>;
@group(0) @binding(1) var<storage, read> matrix_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> matrix_c: array<f32>;
@group(0) @binding(3) var<uniform> dims: Dimensions;

const TILE_SIZE: u32 = 16u;

var<workgroup> tile_a: array<f32, 256>; // 16x16
var<workgroup> tile_b: array<f32, 256>; // 16x16

@compute @workgroup_size(16, 16)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let row = global_id.y;
    let col = global_id.x;
    let local_row = local_id.y;
    let local_col = local_id.x;
    
    var sum: f32 = 0.0;
    
    // Loop over tiles
    let num_tiles = (dims.K + TILE_SIZE - 1u) / TILE_SIZE;
    for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
        // Load tile from A into workgroup memory
        let a_col = t * TILE_SIZE + local_col;
        if (row < dims.M && a_col < dims.K) {
            tile_a[local_row * TILE_SIZE + local_col] = matrix_a[row * dims.K + a_col];
        } else {
            tile_a[local_row * TILE_SIZE + local_col] = 0.0;
        }
        
        // Load tile from B into workgroup memory
        let b_row = t * TILE_SIZE + local_row;
        if (b_row < dims.K && col < dims.N) {
            tile_b[local_row * TILE_SIZE + local_col] = matrix_b[b_row * dims.N + col];
        } else {
            tile_b[local_row * TILE_SIZE + local_col] = 0.0;
        }
        
        workgroupBarrier();
        
        // Compute partial dot product
        for (var k: u32 = 0u; k < TILE_SIZE; k = k + 1u) {
            sum = sum + tile_a[local_row * TILE_SIZE + k] * tile_b[k * TILE_SIZE + local_col];
        }
        
        workgroupBarrier();
    }
    
    // Write result
    if (row < dims.M && col < dims.N) {
        matrix_c[row * dims.N + col] = sum;
    }
}
