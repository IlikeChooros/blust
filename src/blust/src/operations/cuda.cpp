#include <blust/backend/cuda_ops.hpp>

START_BLUST_NAMESPACE

typedef operations::tensor_rref_t tensor_rref_t;

/**
 * @brief Add two tensors and return the result
 */
tensor_rref_t cuda_ops::add(tensor_t a, tensor_t b, bool allocate) 
{
    return a;
    // return M_perform_vector_like(a, b, 1.0, 1.0, M_impl_add, allocate);
}

/**
 * @brief Perform substaction (a - b) and return the result
 */
tensor_rref_t cuda_ops::sub(tensor_t a, tensor_t b, bool allocate) 
{
    return a;
    // return M_perform_vector_like(a, b, 1.0, -1.0, M_impl_add, allocate);
}

/**
 * @brief Caluculate Ri = Ai * b (see hadamard for element-wise multiplication)
 */
tensor_rref_t cuda_ops::mul(tensor_t a, number_t b, bool allocate) 
{
    return a;
    // return M_perform_vector_like(a, a, b, 0.0, M_impl_add, allocate); // c = a * b + a * 0
}

/**
 * @brief Calculate Ri = Ai / b
 */
tensor_rref_t cuda_ops::div(tensor_t a, number_t b, bool allocate) 
{
    return a;
    // return M_perform_vector_like(a, a, 1 / b, 0.0, M_impl_add, allocate);
}

/**
 * @brief Get the hadamard product: Ci = Ai * Bi
 */
tensor_rref_t cuda_ops::hadamard(tensor_t a, tensor_t b, bool allocate)
{
    return a;
    // return M_perform_vector_like(a, b, 0, 0, M_impl_hadamard, allocate);
}

/**
 * @brief Perform matrix multiplication, and return the result
 * @param a the first matrix, with dimensions m x n and in a column-major order
 * @param b the second matrix, with dimensions n x k and in a column-major order
 * @return the result matrix, with dimensions m x k and in a column-major order
 */
tensor_rref_t cuda_ops::mat_mul(tensor_t a, tensor_t b)
{
    M_assert_tensor_dim_mat_mul(a, b);

    const size_t m_rows = a.dim()[0];
    const size_t n_cols = a.dim()[1];
    const size_t k_cols = b.dim()[1];

    auto res = ops_tensor({m_rows, k_cols});
    
    // M_impl_matumul<cpu_ops::matmul_type::see>(
    //     a.data(), n_cols,
    //     b.data(), k_cols,
    //     res.data(), k_cols,
    //     m_rows, n_cols, k_cols
    // );

    return std::move(res);
}

END_BLUST_NAMESPACE