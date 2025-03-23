#pragma once

#include <immintrin.h>
#include <cstring>
#include <string.h>
#include "operations.hpp"

START_BLUST_NAMESPACE

class cpu_ops : public operations
{
    // Preformns c = a * n + b * m
    static void M_impl_add(
        tensor_t::pointer __restrict a, tensor_t::pointer __restrict b, 
        tensor_t::pointer __restrict c, size_t size, number_t n, number_t m
    ) noexcept(true);

    static void M_impl_hadamard(
        tensor_t::pointer __restrict a, tensor_t::pointer __restrict b, 
        tensor_t::pointer __restrict c, size_t size
    ) noexcept(true);

    // Calls M_add with these parameters
    static inline tensor_t M_perform_vector_like(
        tensor_t& a, tensor_t& b, number_t n, number_t m
    );

    static inline tensor_t M_get_res_tensor(tensor_t& a, tensor_t& b);

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