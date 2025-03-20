#pragma once

#include <immintrin.h>
#include "operations.hpp"

START_BLUST_NAMESPACE

class cpu_ops : public operations
{
    // Preformns c = a * n + b * m
    static void M_add(
        tensor_t::pointer a, tensor_t::pointer b, tensor_t::pointer c, 
        size_t size, number_t n = 1.0, number_t m = 1.0) noexcept(true);

    // Calculate dot product
    static void M_dot_prod(
        tensor_t::pointer a, tensor_t::pointer b, tensor_t::pointer c,
        size_t size
    ) noexcept(true);
public:
    cpu_ops() = default;

    tensor_t add(tensor_t, tensor_t);
    tensor_t sub(tensor_t, tensor_t);
    tensor_t mul(tensor_t, number_t);
    tensor_t div(tensor_t, number_t);

    tensor_t hadamard(tensor_t, tensor_t);
    tensor_t mat_mul(tensor_t, tensor_t);
};

END_BLUST_NAMESPACE