#pragma once

#include "BaseLayer.hpp"

START_BLUST_NAMESPACE

// Has fully connected layer of weight to input
class Dense : public BaseDense
{
public:
    Dense(size_t n_outputs) : BaseDense(n_outputs) {}
};

END_BLUST_NAMESPACE