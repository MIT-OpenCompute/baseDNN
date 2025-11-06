#pragma once
#include "tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

// Basic operations
void td_matmul(const td_mat *A, const td_mat *B, td_mat *C);
void td_add(const td_mat *A, const td_mat *B, td_mat *C);
void td_scale(td_mat *A, float c);
void td_transpose(const td_mat *A, td_mat *AT);

// NN Operations
td_mat identity(const td_mat *A);
td_mat td_relu(const td_mat *A);
td_mat td_tanh(const td_mat *A);
td_mat td_sigmoid(const td_mat *A);
td_mat td_softmax_rows(const td_mat *A);

#ifdef __cplusplus
}
#endif