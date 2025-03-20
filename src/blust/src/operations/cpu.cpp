#include <blust/backend/cpu_ops.hpp>

START_BLUST_NAMESPACE

typedef operations::tensor_t tensor_t;
typedef tensor_t::pointer pointer;

typedef union {
    __m256 v;
    float d[4];
} vec4f_t;


void cpu_ops::M_add(
    pointer a_data, pointer b_data, 
    pointer c_data, size_t size, 
    number_t n, number_t m
) noexcept
{
    size_t i = 0;
    // vec4f_t v1, v2;


    // for (;i < size; i++, a_data++, b_data++, c_data++) {
    //     *c_data += *a_data * n + *b_data * m;
    // }

    double res_a0, res_a1, res_a2, res_a3;

    res_a0 = 0.0;
    res_a1 = 0.0;
    res_a2 = 0.0;
    res_a3 = 0.0;

    for (; i < size; i += 4)
    {
        res_a0 = *a_data * n;
        res_a1 = *(a_data + 1) * n;
        res_a2 = *(a_data + 2) * n;
        res_a3 = *(a_data + 3) * n;

        res_a0 += *b_data * m;
        res_a1 += *(b_data + 1) * m;
        res_a2 += *(b_data + 2) * m;
        res_a3 += *(b_data + 3) * m;

        *c_data       = res_a0;
        *(c_data + 1) = res_a1;
        *(c_data + 2) = res_a2;
        *(c_data + 3) = res_a3;
        
        a_data += 4;
        b_data += 4;
        c_data += 4;
    }

    // Add the rest of the elements
    for (;i < size; i++, a_data++, b_data++, c_data++) {
        *c_data += *a_data * n + *b_data * m;
    }
}

/**
 * @brief Add two tensors and return the result
 */
tensor_t cpu_ops::add(tensor_t a, tensor_t b)
{
    M_assert_tensor_same_size(a, b);
    tensor_t res(new number_t[a.size()], a.layout());

    // Perform a tiled addition of the 2 tensors
    M_add(a.data(), b.data(), res.data(), res.size(), 1.0, 1.0);
    return res;
}

/**
 * @brief Perform substaction (a - b) and return the result
 */
tensor_t cpu_ops::sub(tensor_t a, tensor_t b)
{
    M_assert_tensor_same_size(a, b);
    tensor_t res(new number_t[a.size()], a.layout());
    M_add(a.data(), b.data(), res.data(), res.size(), 1.0, -1.0);
    return res;
}

/**
 * @brief Caluculate Ri = Ai * b
 */
tensor_t cpu_ops::mul(tensor_t a, number_t b)
{
    tensor_t res(new number_t[a.size()], a.layout());
    M_add(a.data(), a.data(), res.data(), res.size(), 0.0, b);
    return res;
}

/**
 * @brief Calculate Ri = Ai / b
 */
tensor_t cpu_ops::div(tensor_t a, number_t b)
{
    tensor_t res(new number_t[a.size()], a.layout());
    M_add(a.data(), a.data(), res.data(), res.size(), 0.0, 1 / b);
    return res;
}

/**
 * @brief Get the hadamard product: Ci = Ai * Bi
 */
tensor_t cpu_ops::hadamard(tensor_t a, tensor_t b)
{
    M_assert_tensor_same_size(a, b);

    const auto size = a.size();
    tensor_t res(new number_t[size], a.layout());

    auto a_data = a.data();
    auto b_data = b.data();
    auto c_data = res.data();

    size_t i = 0;
    double res_a0, res_a1, res_a2, res_a3;

    res_a0 = 0.0;
    res_a1 = 0.0;
    res_a2 = 0.0;
    res_a3 = 0.0;

    for (; i < size; i += 4)
    {
        res_a0 = *a_data;
        res_a1 = *(a_data + 1);
        res_a2 = *(a_data + 2);
        res_a3 = *(a_data + 3);

        res_a0 *= *b_data;
        res_a1 *= *(b_data + 1);
        res_a2 *= *(b_data + 2);
        res_a3 *= *(b_data + 3);

        *c_data       = res_a0;
        *(c_data + 1) = res_a1;
        *(c_data + 2) = res_a2;
        *(c_data + 3) = res_a3;
        
        a_data += 4;
        b_data += 4;
        c_data += 4;
    }

    // Add the rest of the elements
    for (;i < size; i++, a_data++, b_data++, c_data++) {
        *c_data += *a_data * *b_data;
    }

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