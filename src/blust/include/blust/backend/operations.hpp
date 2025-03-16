#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <blust/base_types.hpp>
#include <blust/tensor.hpp>

START_BLUST_NAMESPACE

class ops_tensor;


class base_arithmetic_operations 
{
public:
    virtual ops_tensor substract(ops_tensor, ops_tensor) = 0;

    // Any tensor rank operations
    virtual ops_tensor add(ops_tensor, ops_tensor) = 0;
};

class vector_operations
{
public:
    virtual ops_tensor hadamard(ops_tensor, tensor) = 0;
};

class matrix_operations
{
public:
    virtual ops_tensor mat_mul(ops_tensor, ops_tensor) = 0;
};


class operations : public base_arithmetic_operations, public vector_operations, public matrix_operations
{
public:
    operations() = default;
    virtual ~operations() = default;

    // Matrix operations

    // Assumes tensor rank == 2
    virtual ops_tensor mat_mul(tensor, tensor) = 0;
};

END_BLUST_NAMESPACE

