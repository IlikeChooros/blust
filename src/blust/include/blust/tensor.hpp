#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <vector>
#include <algorithm>
#include <numeric>
#include <memory>

#include <cuda.h>

#include "base_types.hpp"
#include "utils.hpp"

START_BLUST_NAMESPACE

// Shape of the tensor
class shape
{
public:
    friend class tensor;

    typedef std::vector<size_t> dim_t;

    shape(std::initializer_list<size_t> dims) 
    {
        m_dims.reserve(dims.size());
        for (auto& d : dims)
            m_dims.push_back(d);
    }

    shape(const shape& other) : m_dims(other.m_dims) {}
    shape& operator=(const shape& other) { m_dims = other.m_dims; return *this; }

    const dim_t& dim() const { return m_dims; }
    size_t rank() const { return m_dims.size(); }
    size_t total() const {
        return std::accumulate(m_dims.begin(), m_dims.end(), 1, std::multiplies<size_t>());
    }

    // Printing shape
    friend std::ostream& operator<<(std::ostream& out, const shape& s)
    {
        out << "rank=" << s.rank() << " dim=";
        for (size_t i = 0; i < s.rank(); i++)
        {
            out << s.m_dims[i];
            if (i != s.rank() - 1)
                out << 'x';
        }
        return out;
    }

private:
    dim_t m_dims;
};


// Main tensor class, can either hold heap memory buffer, or gpu memory pointer
class tensor 
{
public:
    typedef CUdeviceptr cu_pointer;
    typedef number_t* pointer;
    typedef const pointer const_pointer;
    typedef union { CUdeviceptr cu_ptr; pointer data; } internal_data;
    enum class data_type { buffer = 0, cuda = 1 };

    tensor(shape dim, number_t init = 0.0) 
    : m_shape(dim)

    {
        auto size = dim.total();
        m_tensor.data   = new number_t[size]{};
        if (init != 0.0)
            std::fill(m_tensor.data, m_tensor.data + size, init);
        m_data_type     = data_type::buffer;
    }

    tensor(const tensor& t) = default;
    tensor(tensor&& t) = default;

    tensor(cu_pointer cu_ptr, shape dim) : m_shape(dim) 
    {
        m_tensor.cu_ptr = cu_ptr;
        m_data_type     = data_type::cuda;
    }

    tensor(pointer data, shape dim) : m_shape(dim) 
    {
        m_tensor.data   = data;
        m_data_type     = data_type::buffer;
    }

    ~tensor() 
    {
        if (m_data_type == data_type::buffer) {
            delete [] m_tensor.data;
            m_tensor.data = nullptr;
        }
    }

    shape::dim_t dim() const { return m_shape.dim(); }
    size_t rank() const { return m_shape.rank(); }

    cu_pointer cu_data() const { return m_tensor.cu_ptr; }
    pointer data() { return m_tensor.data; }
    const_pointer data() const { return m_tensor.data; }

    // Release the buffer, should be wrapped in a unique pointer with array type
    pointer release() { return M_release_t<pointer>(); }
    cu_pointer cu_release() { return M_release_t<cu_pointer>(); }

    // Print the matrix to output stream
    friend std::ostream& operator<<(std::ostream& out, const tensor& t)
    {
        out << "<dtype=" << utils::TypeName<number_t>() << ", " << t.m_shape << ">\n";

        // print the buffer
        if (t.m_data_type == data_type::buffer)
        {
            auto rank = t.rank();
            if (rank >= 1)
                t.M_print_tensor(t, out, rank);
        }
        
        return out;
    }

private:
    shape m_shape;
    internal_data m_tensor;
    data_type m_data_type;

    // Print the tensor recursively, to given output stream, rank = t.rank(), index = 0, offset = 0
    static void M_print_tensor(const tensor& t, std::ostream& out, size_t rank, size_t index = 0, size_t offset = 0)
    {
        auto end = t.m_shape.m_dims.at(index);

        for (size_t p = 0; p < index; p++)
            out << ' ';
        
        if (rank == 1)
        {            
            out << '[';
            for (size_t i = 0; i < end; i++)
            {
                // print with proper formatting
                if (i == end - 1)
                    out << t.m_tensor.data[offset + i];
                else
                    out << t.m_tensor.data[offset + i] << ", ";   
            }
        }
        else
        {
            out << "[\n";
            for (size_t i = 0; i < end; i++)
            {
                M_print_tensor(t, out, rank - 1, index + 1, offset + i * t.m_shape.m_dims[index + 1]);
            }

            for (size_t p = 0; p < index; p++)
                out << ' ';
        }        
        out << "]\n";
    }


    template <typename T>
    std::enable_if_t<std::is_same_v<T, pointer> || std::is_same_v<T, cu_pointer>, T>
    M_release_t()
    {
        T res;
        if constexpr (std::is_same_v<T, pointer>) { res = m_tensor.data; res = nullptr; }
        else { res = m_tensor.cu_ptr; res = 0; }
        return res;
    }
};


END_BLUST_NAMESPACE

#endif //TENSOR_HPP
