#include <blust/namespaces.hpp>
#include <blust/utils.hpp>
#include <blust/error.hpp>

#include <vector>
#include <iostream>
#include <memory>

START_BLUST_NAMEPSPACE

class shape2D
{
public:
    size_t x, y;
    shape2D() : x(0), y(0) {}

    /**
     * @brief Set the shape, (y, x) (rows, columns, in matrix)
     */
    shape2D(size_t x, size_t y) : x(x), y(y) {}
    shape2D(const shape2D& other) {*this = other;}

    shape2D& operator=(const shape2D& other) 
    {
        this->x = other.x; this->y = other.y;
        return *this;
    }
};

template <typename dtype>
class matrix
{
public:
    typedef dtype* pointer_t;

    matrix() = default;
    matrix(const matrix& other) { this->operator=<dtype>(other); }
    matrix(shape2D shape, dtype init_val = 0)
    {
        static_assert(std::is_arithmetic<dtype>(), 
            "Template parameter in matrix must be an arithmetic type (int, float, double, etc.)");
        build(shape, init_val);
    }

    matrix(std::initializer_list<std::vector<dtype>> list)
    {
        M_alloc_buffer({list.size(), list.begin()->size()});

        size_t r = 0;
        for(auto& v : list)
        {
            for (size_t i = 0; i < v.size(); i++)
                (*this)(r, i) = v[i];
            r++;
        }
    }

    /**
     * @brief Copy operator
     */
    template <typename T>
    matrix& operator=(const matrix<T>& other)
    {
        if (!(m_cols == other.cols() && m_rows == other.rows()))
            M_alloc_buffer({other.rows(), other.cols()});
        
        if constexpr (std::is_same<dtype, T>())
        {
            if (data() != other.data())
                std::copy_n(other.data(), other.size(), data());
        }
        else
        {
            std::copy_n(other.data(), other.size(), data());
        }
        return *this;
    }

    /**
     * @brief Create matrix R x C
     * @param r n rows
     * @param c n cols
     */
    inline void build(shape2D shape, dtype init = 0)
    {
        M_alloc_buffer(shape);
        for (size_t i = 0; i < size(); ++i)
            m_matrix[i] = init;
    }

    // Get the total size of the buffer
    inline size_t size() const { return m_rows * m_cols; }

    // Get number of rows in a matrix
    inline size_t rows() const { return m_rows; }

    // Get number of columns
    inline size_t cols() const { return m_cols; }

    // Get dimensions of a matrix
    inline shape2D dim() const { return {rows(), cols()}; }

    // Get the raw pointer
    inline pointer_t data() const { return m_matrix.get(); }

    // Get transposed matrix 
    matrix T() 
    {
        matrix m({m_cols, m_rows});
        const auto s = size();
        for (size_t n = 0; n < s; ++n)
        {
            int i = n / m_rows;
            int j = n % m_rows;
            m.m_matrix[n] = m_matrix[m_cols*j + i];
        }
        return m;
    }

    // Get the value at (row, column)
    dtype& operator()(size_t r, size_t c) const { return m_matrix[r * m_cols + c]; }
    dtype& operator()(size_t i) const { return m_matrix[i]; }

    // Get the whole row as vector
    std::vector<dtype> operator[](size_t r) 
    { 
        return std::vector<dtype>(
            m_matrix.get() + (r * m_cols), 
            m_matrix.get() + ((r + 1) * m_cols)); 
    }

    // Compare the matrices
    template <typename t>
    friend inline bool operator==(const matrix& rhs, const matrix<t>& lhs)
    {
        if (!(rhs.rows() == lhs.rows() && rhs.cols() == lhs.cols()))
            return false;
        
        auto rb = lhs.m_matrix.get(), 
             re = lhs.m_matrix.get() + lhs.size();
        auto lb = rhs.m_matrix.get();
        return std::equal(rb, re, lb);
    }

    // Multiplication of 2 matrices
    template <typename T>
    friend matrix<dtype> operator*(matrix<dtype>& lhs, matrix<T>& rhs) { return lhs._multip(rhs); }

    // Multiplication of matrix and vector (for simplification, vector is used as if it was vertical)
    // Resulting in vector (also vertical), of size matrix.rows (should be a matrix of dimensions: matrix.rows x 1) 
    template <typename T>
    friend std::vector<dtype> operator*(matrix<dtype>& lhs, std::vector<T>& rhs) { return lhs._multip_v<T, true>(rhs); }

    template <typename T>
    friend std::vector<dtype> operator*(std::vector<T>& lhs, matrix<dtype>& rhs) { return rhs._multip_v<T, false>(lhs); }

    // Multiply matrix by a scalar
    friend matrix<dtype> operator*(matrix<dtype>& lhs, double d) { return lhs._multip_k(d); }
    friend matrix<dtype> operator*(double d, matrix<dtype>& rhs) { return rhs._multip_k(d); }
    friend matrix<dtype> operator*(matrix<dtype>& lhs, int d) { return lhs._multip_k(d); }
    friend matrix<dtype> operator*(int d, matrix<dtype>& rhs) { return rhs._multip_k(d); }

    // Print the matrix to output stream
    friend std::ostream& operator<<(std::ostream& out, const matrix& m)
    {
        out << "<dtype=" << utils::TypeName<dtype>() << ">\n";
        for (size_t r = 0; r < m.rows(); ++r)
        {
            out << '[';
            for (size_t c = 0; c < m.cols(); ++c)
            {
                // compare the addresses 
                if (c == m.cols() - 1)
                    out << m(r, c);
                else
                    out << m(r, c) << ", ";
            }
            out << "]\n";
        }
        
        return out;
    }

private:

    std::unique_ptr<dtype[]> m_matrix;
    size_t m_rows;
    size_t m_cols;


    // Set the internal size and reallocate the buffer
    void M_alloc_buffer(shape2D shape)
    {
        m_rows = shape.x;
        m_cols = shape.y;
        m_matrix.reset(new dtype[size()]);
    }

    // dot product of given vectors, assumes the input is correct (v1.size == v2.size)
    template<typename T>
    dtype dot_product(std::vector<dtype>& v1, std::vector<T>& v2)
    {
        const size_t n = v1.size();
        dtype dot      = 0;
        size_t i       = 0;

        // Unrolled
        for (i = 0; i <= n - 4; i += 4)
        {
            dot += (v1[i]     * v2[i] +
                    v1[i + 1] * v2[i + 1] +
                    v1[i + 2] * v2[i + 2] +
                    v1[i + 3] * v2[i + 3]
            );
        }

        for (; i < n; i++)
            dot += v1[i] * v2[i];

        return dot;
    }

    // Optimized vector multiplication
    template <typename T, bool MatrixFirst>
    std::vector<dtype> _multip_v(std::vector<T>& v)
    {
        if constexpr (MatrixFirst)
        {
            // Assert correct sizes
            if (!(cols() == v.size()))
                throw InvalidMatrixSize({1, v.size()}, {1, cols()});
        
            const size_t n_rows = rows();
            std::vector<dtype> result(n_rows, 0);

            // Calculate the dot product for each row
            for (size_t r = 0; r < n_rows; r++)
            {
                auto row  = (*this)[r];
                result[r] = dot_product(row, v);
            }
            return result;
        }
        else
        {
            if (!(v.size() == rows()))
                throw InvalidMatrixSize({v.size(), 1}, {rows(), 1});
            
            const size_t n_cols = cols();
            std::vector<dtype> result(n_cols, 0);
            auto transp = this->T();

            // Calculate the dot product for each row
            for (size_t c = 0; c < n_cols; c++)
            {
                auto col  = transp[c];
                result[c] = dot_product(col, v);
            }
            return result;
        }
    }

    /**
     * @brief Multiply matrices.
     * @throw May throw `InvalidMatrixSize` if this->cols() != m.rows()
     * @return Product matix (rows() x m.cols())
     */
    template <typename t>
    matrix _multip(matrix<t>& m)
    {
        if (!(cols() == m.rows()))
            throw InvalidMatrixSize({m.rows(), m.cols()}, {cols(), m.cols()});
        
        const size_t m_rows = m.rows(),
                     m_cols = m.cols(),
                     n_rows = rows();
        
        matrix ret({n_rows, m_cols}); 

        // re-order
        for (size_t r1 = 0; r1 < n_rows; r1++) // go through the rows of 1st matrix
            for(size_t k = 0; k < m_rows; ++k) // reorder, go through the rows of 2nd matrix
                for(size_t c2 = 0; c2 < m_cols; c2++) // loop through the columns of 2nd matrix
                    ret(r1, c2) += (*this)(r1, k) * (m(k, c2)); // dot product
        
        return ret;
    }

    template<typename t>
    matrix _multip_tiles(matrix<t>& m)
    {
        constexpr auto cache_size = 16 * (1 << 10);
        constexpr auto block_size = (cache_size / 2 / sizeof(double)) >> 5; 

        if (!(cols() == m.rows()))
            throw InvalidMatrixSize({m.rows(), m.cols()}, {cols(), m.cols()});
        matrix ret({rows(), m.cols()});

        return ret;
    }

    // Multiply the matrix by a scalar
    template <typename t>
    matrix _multip_k(t k)
    {
        static_assert(std::is_arithmetic<t>(), "Given type must be arithmetic (int, double, float, etc.)");

        matrix m = *this;

        for (size_t r = 0; r < rows(); r++)
            for (size_t c = 0; c < cols(); c++)
                m(r, c) *= k;
            
        return m;
    }
};


END_BLUST_NAMESPACE