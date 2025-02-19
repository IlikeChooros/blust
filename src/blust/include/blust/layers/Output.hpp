#pragma once

#include "BaseLayer.hpp"

START_BLUST_NAMESPACE

class Output : public BaseDense
{
public:
    Output(size_t n_outputs) : BaseDense(n_outputs) {}

    // Allocate memory for matrices, set the activation and error function
    void build(shape2D input_shape, activations act, error_funcs err = mean_squared_error)
    {
        BaseDense::build(input_shape, act);
        m_func_error = get_error_function(err);
    }

    number_t cost(matrix_t& expected)
    {
        return m_func_error(m_activations, expected);
    }

private:
    base_error_func_t m_func_error;
};

END_BLUST_NAMESPACE;