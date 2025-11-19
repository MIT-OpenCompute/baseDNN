#include "../../include/ops.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#define EPSILON 1e-4f
#define ASSERT_FLOAT_EQ(a, b) assert(fabsf((a) - (b)) < EPSILON)
#define TEST(name) void test_##name()
#define RUN_TEST(name) do { printf("Running %s...\n", #name); test_##name(); printf("  PASSED\n"); } while(0)

// ====================================================
// Elementwise Operations Tests
// ====================================================

TEST(tensor_add_same_shape) {
    size_t shape[] = {2, 3};
    Tensor *a = tensor_create(shape, 2);
    Tensor *b = tensor_create(shape, 2);
    
    for (size_t i = 0; i < 6; i++) {
        a->data[i] = (float)i;
        b->data[i] = (float)(i + 1);
    }
    
    Tensor *c = tensor_add(a, b);
    
    assert(c != NULL);
    for (size_t i = 0; i < 6; i++) {
        ASSERT_FLOAT_EQ(c->data[i], (float)(2 * i + 1));
    }
    
    tensor_free(a);
    tensor_free(b);
    tensor_free(c);
}

TEST(tensor_add_with_broadcast) {
    size_t shape_a[] = {2, 3};
    size_t shape_b[] = {3};
    
    Tensor *a = tensor_create(shape_a, 2);
    Tensor *b = tensor_create(shape_b, 1);
    
    for (size_t i = 0; i < 6; i++) a->data[i] = 1.0f;
    for (size_t i = 0; i < 3; i++) b->data[i] = (float)i;
    
    Tensor *c = tensor_add(a, b);
    
    assert(c != NULL);
    ASSERT_FLOAT_EQ(c->data[0], 1.0f); // 1 + 0
    ASSERT_FLOAT_EQ(c->data[1], 2.0f); // 1 + 1
    ASSERT_FLOAT_EQ(c->data[2], 3.0f); // 1 + 2
    ASSERT_FLOAT_EQ(c->data[3], 1.0f); // 1 + 0
    ASSERT_FLOAT_EQ(c->data[4], 2.0f); // 1 + 1
    ASSERT_FLOAT_EQ(c->data[5], 3.0f); // 1 + 2
    
    tensor_free(a);
    tensor_free(b);
    tensor_free(c);
}

TEST(tensor_sub) {
    size_t shape[] = {2, 2};
    Tensor *a = tensor_create(shape, 2);
    Tensor *b = tensor_create(shape, 2);
    
    a->data[0] = 5.0f; a->data[1] = 3.0f;
    a->data[2] = 7.0f; a->data[3] = 2.0f;
    
    b->data[0] = 2.0f; b->data[1] = 1.0f;
    b->data[2] = 3.0f; b->data[3] = 1.0f;
    
    Tensor *c = tensor_sub(a, b);
    
    assert(c != NULL);
    ASSERT_FLOAT_EQ(c->data[0], 3.0f);
    ASSERT_FLOAT_EQ(c->data[1], 2.0f);
    ASSERT_FLOAT_EQ(c->data[2], 4.0f);
    ASSERT_FLOAT_EQ(c->data[3], 1.0f);
    
    tensor_free(a);
    tensor_free(b);
    tensor_free(c);
}

TEST(tensor_mul) {
    size_t shape[] = {2, 2};
    Tensor *a = tensor_create(shape, 2);
    Tensor *b = tensor_create(shape, 2);
    
    a->data[0] = 2.0f; a->data[1] = 3.0f;
    a->data[2] = 4.0f; a->data[3] = 5.0f;
    
    b->data[0] = 1.5f; b->data[1] = 2.0f;
    b->data[2] = 0.5f; b->data[3] = 1.0f;
    
    Tensor *c = tensor_mul(a, b);
    
    assert(c != NULL);
    ASSERT_FLOAT_EQ(c->data[0], 3.0f);
    ASSERT_FLOAT_EQ(c->data[1], 6.0f);
    ASSERT_FLOAT_EQ(c->data[2], 2.0f);
    ASSERT_FLOAT_EQ(c->data[3], 5.0f);
    
    tensor_free(a);
    tensor_free(b);
    tensor_free(c);
}

// ====================================================
// Linear Algebra Tests
// ====================================================

TEST(tensor_matmul_2d_2d) {
    size_t shape_a[] = {2, 3};
    size_t shape_b[] = {3, 2};
    
    Tensor *a = tensor_create(shape_a, 2);
    Tensor *b = tensor_create(shape_b, 2);
    
    // A = [[1, 2, 3],
    //      [4, 5, 6]]
    a->data[0] = 1.0f; a->data[1] = 2.0f; a->data[2] = 3.0f;
    a->data[3] = 4.0f; a->data[4] = 5.0f; a->data[5] = 6.0f;
    
    // B = [[1, 2],
    //      [3, 4],
    //      [5, 6]]
    b->data[0] = 1.0f; b->data[1] = 2.0f;
    b->data[2] = 3.0f; b->data[3] = 4.0f;
    b->data[4] = 5.0f; b->data[5] = 6.0f;
    
    Tensor *c = tensor_matmul(a, b);
    
    assert(c != NULL);
    assert(c->ndim == 2);
    assert(c->shape[0] == 2);
    assert(c->shape[1] == 2);
    
    // C = [[22, 28],
    //      [49, 64]]
    ASSERT_FLOAT_EQ(c->data[0], 22.0f);
    ASSERT_FLOAT_EQ(c->data[1], 28.0f);
    ASSERT_FLOAT_EQ(c->data[2], 49.0f);
    ASSERT_FLOAT_EQ(c->data[3], 64.0f);
    
    tensor_free(a);
    tensor_free(b);
    tensor_free(c);
}

TEST(tensor_matmul_2d_1d) {
    size_t shape_a[] = {2, 3};
    size_t shape_b[] = {3};
    
    Tensor *a = tensor_create(shape_a, 2);
    Tensor *b = tensor_create(shape_b, 1);
    
    a->data[0] = 1.0f; a->data[1] = 2.0f; a->data[2] = 3.0f;
    a->data[3] = 4.0f; a->data[4] = 5.0f; a->data[5] = 6.0f;
    
    b->data[0] = 1.0f; b->data[1] = 2.0f; b->data[2] = 3.0f;
    
    Tensor *c = tensor_matmul(a, b);
    
    assert(c != NULL);
    assert(c->ndim == 1);
    assert(c->shape[0] == 2);
    
    ASSERT_FLOAT_EQ(c->data[0], 14.0f); // 1*1 + 2*2 + 3*3
    ASSERT_FLOAT_EQ(c->data[1], 32.0f); // 4*1 + 5*2 + 6*3
    
    tensor_free(a);
    tensor_free(b);
    tensor_free(c);
}

TEST(tensor_matmul_1d_1d) {
    size_t shape[] = {3};
    
    Tensor *a = tensor_create(shape, 1);
    Tensor *b = tensor_create(shape, 1);
    
    a->data[0] = 1.0f; a->data[1] = 2.0f; a->data[2] = 3.0f;
    b->data[0] = 4.0f; b->data[1] = 5.0f; b->data[2] = 6.0f;
    
    Tensor *c = tensor_matmul(a, b);
    
    assert(c != NULL);
    assert(c->ndim == 1);
    assert(c->size == 1);
    
    ASSERT_FLOAT_EQ(c->data[0], 32.0f); // 1*4 + 2*5 + 3*6
    
    tensor_free(a);
    tensor_free(b);
    tensor_free(c);
}

TEST(tensor_transpose2d) {
    size_t shape[] = {2, 3};
    Tensor *a = tensor_create(shape, 2);
    
    a->data[0] = 1.0f; a->data[1] = 2.0f; a->data[2] = 3.0f;
    a->data[3] = 4.0f; a->data[4] = 5.0f; a->data[5] = 6.0f;
    
    Tensor *b = tensor_transpose2d(a);
    
    assert(b != NULL);
    assert(b->ndim == 2);
    assert(b->shape[0] == 3);
    assert(b->shape[1] == 2);
    
    ASSERT_FLOAT_EQ(b->data[0], 1.0f); ASSERT_FLOAT_EQ(b->data[1], 4.0f);
    ASSERT_FLOAT_EQ(b->data[2], 2.0f); ASSERT_FLOAT_EQ(b->data[3], 5.0f);
    ASSERT_FLOAT_EQ(b->data[4], 3.0f); ASSERT_FLOAT_EQ(b->data[5], 6.0f);
    
    tensor_free(a);
    tensor_free(b);
}

// ====================================================
// Activation Function Tests
// ====================================================

TEST(tensor_relu) {
    size_t shape[] = {4};
    Tensor *a = tensor_create(shape, 1);
    
    a->data[0] = -2.0f;
    a->data[1] = -0.5f;
    a->data[2] = 0.0f;
    a->data[3] = 1.5f;
    
    Tensor *b = tensor_relu(a);
    
    assert(b != NULL);
    ASSERT_FLOAT_EQ(b->data[0], 0.0f);
    ASSERT_FLOAT_EQ(b->data[1], 0.0f);
    ASSERT_FLOAT_EQ(b->data[2], 0.0f);
    ASSERT_FLOAT_EQ(b->data[3], 1.5f);
    
    tensor_free(a);
    tensor_free(b);
}

TEST(tensor_sigmoid) {
    size_t shape[] = {3};
    Tensor *a = tensor_create(shape, 1);
    
    a->data[0] = 0.0f;
    a->data[1] = 1.0f;
    a->data[2] = -1.0f;
    
    Tensor *b = tensor_sigmoid(a);
    
    assert(b != NULL);
    ASSERT_FLOAT_EQ(b->data[0], 0.5f);
    ASSERT_FLOAT_EQ(b->data[1], 1.0f / (1.0f + expf(-1.0f)));
    ASSERT_FLOAT_EQ(b->data[2], 1.0f / (1.0f + expf(1.0f)));
    
    tensor_free(a);
    tensor_free(b);
}

TEST(tensor_tanh) {
    size_t shape[] = {3};
    Tensor *a = tensor_create(shape, 1);
    
    a->data[0] = 0.0f;
    a->data[1] = 1.0f;
    a->data[2] = -1.0f;
    
    Tensor *b = tensor_tanh(a);
    
    assert(b != NULL);
    ASSERT_FLOAT_EQ(b->data[0], 0.0f);
    ASSERT_FLOAT_EQ(b->data[1], tanhf(1.0f));
    ASSERT_FLOAT_EQ(b->data[2], tanhf(-1.0f));
    
    tensor_free(a);
    tensor_free(b);
}

TEST(tensor_softmax) {
    size_t shape[] = {3};
    Tensor *a = tensor_create(shape, 1);
    
    a->data[0] = 1.0f;
    a->data[1] = 2.0f;
    a->data[2] = 3.0f;
    
    Tensor *b = tensor_softmax(a);
    
    assert(b != NULL);
    
    // Sum should be 1.0
    float sum = 0.0f;
    for (size_t i = 0; i < 3; i++) {
        sum += b->data[i];
    }
    ASSERT_FLOAT_EQ(sum, 1.0f);
    
    // Values should be positive
    for (size_t i = 0; i < 3; i++) {
        assert(b->data[i] > 0.0f);
    }
    
    tensor_free(a);
    tensor_free(b);
}

TEST(tensor_softmax_2d) {
    size_t shape[] = {2, 3};
    Tensor *a = tensor_create(shape, 2);
    
    a->data[0] = 1.0f; a->data[1] = 2.0f; a->data[2] = 3.0f;
    a->data[3] = 1.0f; a->data[4] = 1.0f; a->data[5] = 1.0f;
    
    Tensor *b = tensor_softmax(a);
    
    assert(b != NULL);
    
    // Each row should sum to 1.0
    float sum1 = b->data[0] + b->data[1] + b->data[2];
    float sum2 = b->data[3] + b->data[4] + b->data[5];
    
    ASSERT_FLOAT_EQ(sum1, 1.0f);
    ASSERT_FLOAT_EQ(sum2, 1.0f);
    
    // Second row should have equal values
    ASSERT_FLOAT_EQ(b->data[3], 1.0f/3.0f);
    ASSERT_FLOAT_EQ(b->data[4], 1.0f/3.0f);
    ASSERT_FLOAT_EQ(b->data[5], 1.0f/3.0f);
    
    tensor_free(a);
    tensor_free(b);
}

// ====================================================
// Loss Function Tests
// ====================================================

TEST(tensor_mse) {
    size_t shape[] = {4};
    Tensor *pred = tensor_create(shape, 1);
    Tensor *target = tensor_create(shape, 1);
    
    pred->data[0] = 1.0f; pred->data[1] = 2.0f;
    pred->data[2] = 3.0f; pred->data[3] = 4.0f;
    
    target->data[0] = 1.5f; target->data[1] = 2.5f;
    target->data[2] = 2.5f; target->data[3] = 4.5f;
    
    Tensor *loss = tensor_mse(pred, target);
    
    assert(loss != NULL);
    assert(loss->size == 1);
    
    // MSE = ((0.5)^2 + (0.5)^2 + (0.5)^2 + (0.5)^2) / 4 = 1.0 / 4 = 0.25
    ASSERT_FLOAT_EQ(loss->data[0], 0.25f);
    
    tensor_free(pred);
    tensor_free(target);
    tensor_free(loss);
}

TEST(tensor_cross_entropy) {
    size_t shape[] = {3};
    Tensor *pred = tensor_create(shape, 1);
    Tensor *target = tensor_create(shape, 1);
    
    pred->data[0] = 0.7f; pred->data[1] = 0.2f; pred->data[2] = 0.1f;
    target->data[0] = 1.0f; target->data[1] = 0.0f; target->data[2] = 0.0f;
    
    Tensor *loss = tensor_cross_entropy(pred, target);
    
    assert(loss != NULL);
    assert(loss->size == 1);
    assert(loss->data[0] > 0.0f);
    
    tensor_free(pred);
    tensor_free(target);
    tensor_free(loss);
}

TEST(tensor_binary_cross_entropy) {
    size_t shape[] = {4};
    Tensor *pred = tensor_create(shape, 1);
    Tensor *target = tensor_create(shape, 1);
    
    pred->data[0] = 0.9f; pred->data[1] = 0.1f;
    pred->data[2] = 0.8f; pred->data[3] = 0.3f;
    
    target->data[0] = 1.0f; target->data[1] = 0.0f;
    target->data[2] = 1.0f; target->data[3] = 0.0f;
    
    Tensor *loss = tensor_binary_cross_entropy(pred, target);
    
    assert(loss != NULL);
    assert(loss->size == 1);
    assert(loss->data[0] > 0.0f);
    
    tensor_free(pred);
    tensor_free(target);
    tensor_free(loss);
}

// ====================================================
// Slice Tests
// ====================================================

TEST(tensor_slice) {
    size_t shape[] = {4, 3};
    Tensor *a = tensor_create(shape, 2);
    
    for (size_t i = 0; i < 12; i++) {
        a->data[i] = (float)i;
    }
    
    Tensor *slice = tensor_slice(a, 1, 3);
    
    assert(slice != NULL);
    assert(slice->shape[0] == 2);
    assert(slice->shape[1] == 3);
    assert(slice->size == 6);
    assert(slice->owns_data == 0);
    
    ASSERT_FLOAT_EQ(slice->data[0], 3.0f);
    ASSERT_FLOAT_EQ(slice->data[5], 8.0f);
    
    tensor_free(slice);
    tensor_free(a);
}

// ====================================================
// Gradient Tests
// ====================================================

TEST(backward_add) {
    size_t shape[] = {2, 2};
    Tensor *a = tensor_create(shape, 2);
    Tensor *b = tensor_create(shape, 2);
    
    a->data[0] = 1.0f; a->data[1] = 2.0f;
    a->data[2] = 3.0f; a->data[3] = 4.0f;
    
    b->data[0] = 5.0f; b->data[1] = 6.0f;
    b->data[2] = 7.0f; b->data[3] = 8.0f;
    
    tensor_set_requires_grad(a, 1);
    tensor_set_requires_grad(b, 1);
    
    Tensor *c = tensor_add(a, b);
    assert(c->requires_grad == 1);
    
    tensor_backward(c);
    
    assert(a->grad != NULL);
    assert(b->grad != NULL);
    
    for (size_t i = 0; i < 4; i++) {
        ASSERT_FLOAT_EQ(a->grad[i], 1.0f);
        ASSERT_FLOAT_EQ(b->grad[i], 1.0f);
    }
    
    tensor_free(a);
    tensor_free(b);
    tensor_free(c);
}

TEST(backward_mul) {
    size_t shape[] = {2};
    Tensor *a = tensor_create(shape, 1);
    Tensor *b = tensor_create(shape, 1);
    
    a->data[0] = 2.0f; a->data[1] = 3.0f;
    b->data[0] = 4.0f; b->data[1] = 5.0f;
    
    tensor_set_requires_grad(a, 1);
    tensor_set_requires_grad(b, 1);
    
    Tensor *c = tensor_mul(a, b);
    tensor_backward(c);
    
    assert(a->grad != NULL);
    assert(b->grad != NULL);
    
    ASSERT_FLOAT_EQ(a->grad[0], 4.0f); // grad w.r.t. a[0] = b[0]
    ASSERT_FLOAT_EQ(a->grad[1], 5.0f); // grad w.r.t. a[1] = b[1]
    ASSERT_FLOAT_EQ(b->grad[0], 2.0f); // grad w.r.t. b[0] = a[0]
    ASSERT_FLOAT_EQ(b->grad[1], 3.0f); // grad w.r.t. b[1] = a[1]
    
    tensor_free(a);
    tensor_free(b);
    tensor_free(c);
}

TEST(backward_relu) {
    size_t shape[] = {4};
    Tensor *a = tensor_create(shape, 1);
    
    a->data[0] = -2.0f;
    a->data[1] = -0.5f;
    a->data[2] = 0.0f;
    a->data[3] = 1.5f;
    
    tensor_set_requires_grad(a, 1);
    
    Tensor *b = tensor_relu(a);
    tensor_backward(b);
    
    assert(a->grad != NULL);
    
    ASSERT_FLOAT_EQ(a->grad[0], 0.0f);
    ASSERT_FLOAT_EQ(a->grad[1], 0.0f);
    ASSERT_FLOAT_EQ(a->grad[2], 0.0f);
    ASSERT_FLOAT_EQ(a->grad[3], 1.0f);
    
    tensor_free(a);
    tensor_free(b);
}

// ====================================================
// Main Test Runner
// ====================================================

int main() {
    printf("=== Running Ops Tests ===\n\n");
    
    // Elementwise operations
    RUN_TEST(tensor_add_same_shape);
    RUN_TEST(tensor_add_with_broadcast);
    RUN_TEST(tensor_sub);
    RUN_TEST(tensor_mul);
    
    // Linear algebra
    RUN_TEST(tensor_matmul_2d_2d);
    RUN_TEST(tensor_matmul_2d_1d);
    RUN_TEST(tensor_matmul_1d_1d);
    RUN_TEST(tensor_transpose2d);
    
    // Activation functions
    RUN_TEST(tensor_relu);
    RUN_TEST(tensor_sigmoid);
    RUN_TEST(tensor_tanh);
    RUN_TEST(tensor_softmax);
    RUN_TEST(tensor_softmax_2d);
    
    // Loss functions
    RUN_TEST(tensor_mse);
    RUN_TEST(tensor_cross_entropy);
    RUN_TEST(tensor_binary_cross_entropy);
    
    // Slice
    RUN_TEST(tensor_slice);
    
    // Gradients
    RUN_TEST(backward_add);
    RUN_TEST(backward_mul);
    RUN_TEST(backward_relu);
    
    printf("\n=== All Ops Tests Passed! ===\n");
    return 0;
}
