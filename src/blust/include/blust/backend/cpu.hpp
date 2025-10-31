#pragma once

#include "base_backend.hpp"
#include "cpu_ops.hpp"

#include <cstring>

START_BLUST_NAMESPACE

// CPU backend for matrix operations
class cpu_backend : public base_backend
{
	static npointer_t M_get_d_activations(const_npointer_t outputs, size_t n, activations act_type);
public:
	cpu_backend() = default;

	cpu_backend(const cpu_backend& other) = delete;
	cpu_backend(cpu_backend&& other) = delete;

	// Return the name of the backend 'cpu'
	const char* get_name() override { return "cpu"; }

	void relu(npointer_t input, npointer_t result, size_t n) override;
	void sigmoid(npointer_t input, npointer_t result, size_t n) override;
	void softmax(npointer_t input, npointer_t result, size_t n) override;
	void none(npointer_t input, npointer_t result, size_t n) override {
		memcpy(result, input, n * sizeof(number_t));
	}

	void backprop_dense_output(
		number_t *outputs, number_t *expected, activations act_type,
		number_t *parial_deriv, shape2D output_shape, size_t n_batch) override;

	void backprop_hidden_dense(
		number_t *d_weights, number_t *d_biases, activations act_type,
		number_t *d_prev_activations, number_t *weights, number_t *inputs,
		number_t *prev_d_activations, number_t *prev_weights,
		size_t n_weights, size_t n_prev_activations,
		size_t n_inputs, size_t n_batch) override;

	void vector_add(number_t* res, number_t* mat1, number_t* mat2, size_t N) override;
	void vector_sub(number_t* res, number_t* mat1, number_t* mat2, size_t N) override;
	void vector_scalar_mul(number_t* res, number_t* mat, number_t scalar, size_t N) override;
	void vector_mul_hadamard(number_t* res, number_t* mat1, number_t* mat2, size_t N) override;
	void mat_transpose(number_t* res, number_t* mat, size_t rows, size_t cols) override;
	void mat_mul(number_t* res, number_t* mat1, number_t* mat2, size_t rows1, size_t cols2, size_t rows2) override;
};

END_BLUST_NAMESPACE