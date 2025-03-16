#pragma once

#include <blust/base_types.hpp>

#include <memory>

START_BLUST_NAMESPACE

// Base backend for matrix operations (either cpu or gpu) (only 1 backend can be used at a time)
class base_backend
{
public:
	// Default constructor
	base_backend() = default;
	virtual ~base_backend() = default;

	// Return the name of the backend
	virtual const char* get_name() = 0;

	// Activations
	virtual void relu(npointer_t input, npointer_t result, size_t n) = 0;
	virtual void sigmoid(npointer_t input, npointer_t result, size_t n) = 0;
	virtual void softmax(npointer_t input, npointer_t result, size_t n) = 0;
	virtual void none(npointer_t input, npointer_t result, size_t n) = 0;

	// Backpropagation
	virtual void backprop_dense_output(
		number_t* outputs, number_t* expected, activations act_type,
		number_t* parial_deriv, shape2D output_shape, size_t n_batch) = 0;

	// Prev partial deriv, prev weights, 
	virtual void backprop_hidden_dense(
		number_t* d_weights, number_t* d_biases, activations act_type, number_t* d_prev_activations,
		number_t* weights, number_t* inputs, number_t* prev_d_activations, number_t* prev_weights,
		size_t n_weights, size_t n_prev_activations, size_t n_inputs, size_t n_batch) = 0;


	// Specific functions
	virtual void vector_add(number_t* res, number_t* mat1, number_t* mat2, size_t N) = 0;
	virtual void vector_sub(number_t* res, number_t* mat1, number_t* mat2, size_t N) = 0;
	virtual void vector_mul_hadamard(number_t* res, number_t* mat1, number_t* mat2, size_t N) = 0;
	virtual void vector_scalar_mul(number_t* res, number_t* mat, number_t scalar, size_t N) = 0;
	virtual void mat_transpose(number_t* res, number_t* mat, size_t rows, size_t cols) = 0;
	virtual void mat_mul(number_t* res, number_t* mat1, number_t* mat2, size_t rows1, size_t cols2, size_t rows2) = 0;
};

// Base backend, with memory preallocation (for gpu memory reservation)
class base_memory_backend : public base_backend
{
public:
	// Default constructor
	base_memory_backend() = default;
	virtual ~base_memory_backend() = default;

	// Reserve memory for the given size (on 3 buffers)
	virtual void reserve(size_t size_bytes) = 0;
};

// The backend that is used for all operations (initialized in main.cpp)
static std::unique_ptr<base_backend> g_backend;
constexpr std::unique_ptr<base_backend>& get_backend() { return g_backend; }

END_BLUST_NAMESPACE