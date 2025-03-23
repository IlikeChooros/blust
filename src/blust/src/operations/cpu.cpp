#include <blust/backend/cpu_ops.hpp>

#include <sys/time.h>

START_BLUST_NAMESPACE

typedef operations::tensor_t tensor_t;
typedef tensor_t::pointer pointer;

typedef union {
    __m128 v;
    float d[4];
} vec4f_t;

/**
 * @brief Make the compiler assume that the `data` is n-byte aligned (n = tensor alignment)
 */
constexpr void assume_aligned(pointer data) {
    data = (pointer) __builtin_assume_aligned(data, tensor::alignment);
}

/**
 * @brief Performs c = a * n + b * m, a,b and c must be 16-byte aligned
 */
void cpu_ops::M_impl_add(
    pointer __restrict a_data, pointer __restrict b_data, 
    pointer __restrict c_data, size_t size, 
    number_t n, number_t m
) noexcept
{
    assume_aligned(a_data);
    assume_aligned(b_data);
    assume_aligned(c_data);

    // with -03 and -mavx this is faster
    while (size--) {
        (*c_data++) = (*a_data++) * n + (*b_data++) * m;
    }

    // size_t i = 0;

    // if (size >= 4)
    // {
    //     vec4f_t va, vb, vc, vn, vm;

    //     // Must be aligned
    //     alignas(tensor::alignment) number_t nvec[4] = {n, n, n, n};
    //     alignas(tensor::alignment) number_t mvec[4] = {m, m, m, m};

    //     vn.v = _mm_load_ps(nvec);
    //     vm.v = _mm_load_ps(mvec);

    //     for (; i < size; i += 4)
    //     {
    //         va.v = _mm_load_ps(a_data + i);
    //         vb.v = _mm_load_ps(b_data + i);
    //         vc.v = _mm_add_ps(_mm_mul_ps(va.v, vn.v), _mm_mul_ps(vb.v, vm.v));
    //         _mm_store_ps(c_data + i, vc.v);
    //     }
    // }

    // // Add the rest of the elements
    // for (;i < size; i++) {
    //     (*c_data++) += (*a_data++) * n + (*b_data++) * m;
    // }
}

/**
 * @brief Performs c = a * b, a, b and c must be n-byte aligned (n = tensor alignment), 
 * optimized to use simd instructions
 */
void cpu_ops::M_impl_hadamard(
    pointer __restrict a_data, pointer __restrict b_data, 
    pointer __restrict c_data, size_t size
) noexcept
{
    assume_aligned(a_data);
    assume_aligned(b_data);
    assume_aligned(c_data);

    // with -03 and -mavx this is faster
    while (size--) {
        (*c_data++) = (*a_data++) * (*b_data++);
    }
}

// Get the result tensor, based on the a's and b's operation flags.
// Asserts same size of a and b, then tries to borrow a's or b's buffers
inline tensor_t cpu_ops::M_get_res_tensor(tensor_t& a, tensor_t& b)
{
    // Assert same size
    M_assert_tensor_same_size(a, b);

    // Try to borrow buffers, to avoid redundand memory allocation
    tensor_t res = ops_tensor::M_get_vector_like(a, b);

    // if in chained operation, will use that fact in the function above
    res.set_in_operation(true); 
    return res;
}

/**
 * @brief Calls `M_impl_add` with these parameters, returns the result tensor
 */
inline tensor_t cpu_ops::M_perform_vector_like(tensor_t& a, tensor_t& b, number_t n, number_t m)
{
    auto res = M_get_res_tensor(a, b);
    // calculate the result
    M_impl_add(a.data(), b.data(), res.data(), res.size(), n, m); 
    return res;
}

/**
 * @brief Add two tensors and return the result
 */
tensor_t cpu_ops::add(tensor_t a, tensor_t b) 
{
    return M_perform_vector_like(a, b, 1.0, 1.0);
}

/**
 * @brief Perform substaction (a - b) and return the result
 */
tensor_t cpu_ops::sub(tensor_t a, tensor_t b) 
{
    return M_perform_vector_like(a, b, 1.0, -1.0);
}

/**
 * @brief Caluculate Ri = Ai * b
 */
tensor_t cpu_ops::mul(tensor_t a, number_t b) 
{
    return M_perform_vector_like(a, a, b, 0.0); // c = a * b + a * 0
}

/**
 * @brief Calculate Ri = Ai / b
 */
tensor_t cpu_ops::div(tensor_t a, number_t b) 
{
    return M_perform_vector_like(a, a, 1 / b, 0.0);
}

/**
 * @brief Get the hadamard product: Ci = Ai * Bi
 */
tensor_t cpu_ops::hadamard(tensor_t a, tensor_t b)
{
    tensor_t res = M_get_res_tensor(a, b);
    M_impl_hadamard(a.data(), b.data(), res.data(), res.size());
    return res;
}

/**
 * @brief Perform matrix multiplication, and return the result
 */
tensor_t cpu_ops::mat_mul(tensor_t a, tensor_t b)
{
    return a;
}

END_BLUST_NAMESPACE