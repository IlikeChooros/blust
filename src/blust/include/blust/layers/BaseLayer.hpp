#pragma once

#include <blust/matrix/matrix.hpp>

START_BLUST_NAMEPSPACE

// Types of layers:
// - Dense (input x ouput, should have activation function specified, )
// - Input (input) (not really a layer tho)
// - Output (output, with activation functions)
// - Convolution Layer

// Because the layers will be used as a whole in a model
// I should be able to get the derivatives needed for backpropagation

class BaseLayer
{
public:
    typedef double dtype;

    virtual void build(int input_size);

};


END_BLUST_NAMESPACE