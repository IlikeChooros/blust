#include <blust/backend/cpu.hpp>

START_BLUST_NAMESPACE

// Add the matrices (Cij = Aij + Bij)
void cpu_backend::vector_add(number_t* res, number_t* mat1, number_t* mat2, size_t N)
{
	for (size_t i = 0; i < N; i++)
		res[i] = mat1[i] + mat2[i];
}

// Subtract the matrices (Cij = Aij - Bij)
void cpu_backend::vector_sub(number_t* res, number_t* mat1, number_t* mat2, size_t N)
{
	for (size_t i = 0; i < N; i++)
		res[i] = mat1[i] - mat2[i];
}

// Multiply the matrix by a scalar (k * M)
void cpu_backend::vector_scalar_mul(number_t* res, number_t* mat1, number_t scalar, size_t N)
{
	for (size_t i = 0; i < N; i++)
		res[i] = mat1[i] * scalar;
}

// Hadamard product, element-wise multiplication (Cij = Aij * Bij)
void cpu_backend::vector_mul_hadamard(number_t* res, number_t* mat1, number_t* mat2, size_t N)
{
	for (size_t i = 0; i < N; i++)
		res[i] = mat1[i] * mat2[i];
}

// Transpose the matrix
void cpu_backend::mat_transpose(number_t* res, number_t* mat, size_t rows, size_t cols)
{
	size_t size = rows * cols;
	for (size_t n = 0; n < size; ++n)
	{
		size_t i = n / rows;
		size_t j = n % rows;
		res[n] = mat[cols * j + i];
	}
}

// Typical matrix multiplication
void cpu_backend::mat_mul(number_t* res, number_t* mat1, number_t* mat2, size_t rows1, size_t cols2, size_t rows2)
{
	for (size_t r1 = 0; r1 < rows1; r1++) 
	{
		for (size_t k = 0; k < cols2; ++k)
		{
			number_t sum = 0;
			for (size_t c1 = 0; c1 < rows2; c1++)
				sum += mat1[r1 * rows2 + c1] * mat2[c1 * cols2 + k]; 

			res[r1 * cols2 + k] = sum;
		}
	}
}

END_BLUST_NAMESPACE