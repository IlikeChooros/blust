#pragma once

#include "BaseLayer.hpp"

START_BLUST_NAMESPACE

// Has fully connected layer of weight to input
class Input : public BaseLayer
{
public:
    Input(shape shape) { 
        m_output_shape  = shape; 
        m_output_size   = 0;
        m_inputs_size   = 0;
    }

    Input(const Input& other) : BaseLayer(other)
    {
        m_output_shape = other.m_output_shape;
        m_activations  = other.m_activations;
    }

    Input(Input&& other) : BaseLayer(std::forward<BaseLayer>(other))
    {
        m_output_shape = other.m_output_shape;
        m_activations  = std::move(other.m_activations);
    }

    // Set the `activations`
    tensor_t& feed_forward(tensor_t& inputs) override
    {
        m_activations = inputs;
        return m_activations;
    }
};

END_BLUST_NAMESPACE