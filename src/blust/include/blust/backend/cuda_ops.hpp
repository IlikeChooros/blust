#pragma once

#include "operations.hpp"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas.h>


START_BLUST_NAMESPACE

class cuda_ops : public operations
{
public:

    tensor_rref_t add(tensor_t, tensor_t) override;
    tensor_rref_t sub(tensor_t, tensor_t) override;
    tensor_rref_t mul(tensor_t, number_t) override;
    tensor_rref_t div(tensor_t, number_t) override;

    tensor_rref_t hadamard(tensor_t, tensor_t) override;
    tensor_rref_t mat_mul(tensor_t, tensor_t) override;
};

END_BLUST_NAMESPACE