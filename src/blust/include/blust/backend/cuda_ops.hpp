#pragma once

#include "operations.hpp"

#if ENABLE_CUDA_BACKEND
#   include <cuda.h>
#   include <cuda_runtime_api.h>
#   include <cublas.h>
#endif

START_BLUST_NAMESPACE

class cuda_ops : public operations
{
public:

#if ENABLE_CUDA_BACKEND
    tensor_rref_t add(tensor_t, tensor_t, bool allocate = true) override;
    tensor_rref_t sub(tensor_t, tensor_t, bool allocate = true) override;
    tensor_rref_t mul(tensor_t, number_t, bool allocate = true) override;
    tensor_rref_t div(tensor_t, number_t, bool allocate = true) override;

    tensor_rref_t hadamard(tensor_t, tensor_t, bool allocate = true) override;
    tensor_rref_t mat_mul(tensor_t, tensor_t) override;
    tensor_rref_t transpose(tensor_t) override;
#else 
    // Dummy implementations when CUDA backend is disabled
    tensor_rref_t add(tensor_t a, tensor_t b, bool allocate = true) override { return a; }
    tensor_rref_t sub(tensor_t a, tensor_t b, bool allocate = true) override { return a; }
    tensor_rref_t mul(tensor_t a, number_t b, bool allocate = true) override { return a; }
    tensor_rref_t div(tensor_t a, number_t b, bool allocate = true) override { return a; }
    tensor_rref_t hadamard(tensor_t a, tensor_t b, bool allocate = true) override { return a; }
    tensor_rref_t mat_mul(tensor_t a, tensor_t b) override { return a; }
    tensor_rref_t transpose(tensor_t a) override { return a; }
#endif
};

END_BLUST_NAMESPACE