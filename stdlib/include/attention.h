#ifndef ATTENTION_H
#define ATTENTION_H

#include "../../core/include/tensor.h"

// ====================================================
// Attention Operations
// ====================================================

Tensor* tensor_softmax_row(Tensor *input);
void backward_softmax_row(Tensor *output);

Tensor* tensor_scaled_dot_product_attention(Tensor *Q, Tensor *K, Tensor *V, Tensor *mask);
void backward_scaled_dot_product_attention(Tensor *output);

Tensor* tensor_layer_norm(Tensor *input, Tensor *gamma, Tensor *beta, float eps);
void backward_layer_norm(Tensor *output);

Tensor* tensor_batch_norm(Tensor *input, Tensor *gamma, Tensor *beta, Tensor *running_mean, Tensor *running_var, float momentum, float eps, int training);
void backward_batch_norm(Tensor *output);

// ====================================================
// Attention Layers
// ====================================================

typedef struct MultiHeadedAttentionParams {
    size_t embed_dim;
    size_t num_heads;
    float dropout; 
    int use_bias;
} MultiHeadAttentionParams;

#define MULTIHEADATTENTION(embed_d, num_h, drop, u_b)(LayerConfig){.name="multihead_attention", .params=&(MultiHeadAttentionParams){embed_d, num_h, drop, u_b}}

typedef struct LayerNormParams {
    size_t normalized_shape;
    float eps;
} LayerNormParams;

#define LAYERNORM(norm_shape, eps)(LayerConfig){.name="layer_norm", .params=&(LayerNormParams){norm_shape, eps}}

typedef struct TransformerEncoderParams {
    size_t embed_dim;
    size_t num_heads;
    size_t ff_hidden_dim;
    float dropout;
    float activation_dropout;
} TransformerEncoderParams; 

#define TRANSFORMERENCODER(embed_d, num_h, ff_hidden_d, drop, act_drop)(LayerConfig){.name="transformer_encoder", .params=&(TransformerEncoderParams){embed_d, num_h, ff_hidden_d, drop, act_drop}}

typedef struct PositionalEncodingParams {
    size_t max_len;
    size_t embed_dim;
    float droupout; 
} PositionalEncodingParams;

#define POSITIONALENCODING(max_l, embed_d, drop)(LayerConfig){.name="positional_encoding", .params=&(PositionalEncodingParams){max_l, embed_d, drop}}

typedef struct EmbeddingParams {
    size_t num_embeddings;
    size_t embedding_dim;
} EmbeddingParams;

#define EMBEDDING(num_emb, emb_dim)(LayerConfig){.name="embedding", .params=&(EmbeddingParams){num_emb, emb_dim}}

#endif