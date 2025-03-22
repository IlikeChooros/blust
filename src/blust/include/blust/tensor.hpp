#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <vector>
#include <algorithm>
#include <numeric>
#include <memory>

#include <cuda.h>

#include "utils.hpp"
#include "shape.hpp"

START_BLUST_NAMESPACE


// Main tensor class, can either hold heap memory buffer, or gpu memory pointer
// The buffer is 16-byte aligned
class tensor 
{
public:
    friend class operations;
    friend class cpu_ops;
    friend class ops_tensor;

    typedef CUdeviceptr cu_pointer;
    typedef number_t* pointer;
    typedef const pointer const_pointer;
    typedef union { CUdeviceptr cu_ptr; pointer data; } internal_data;
    enum class data_type { buffer = 1, cuda = 2 };

    // Memory alignment
    static constexpr size_t alignment = 16;

    /**
     * @brief Get total size in bytes, with given alignment
     */
    template <size_t Alignment>
    static constexpr size_t get_bytesize(size_t count) noexcept
    {
        return ((count * sizeof(number_t) + Alignment - 1) / Alignment ) * Alignment;
    }

    /**
     * @brief Get total size in bytes
     */
    static constexpr size_t get_bytesize(size_t count) noexcept
    {
        return get_bytesize<alignment>(count);
    }

    /**
     * @brief Allocates memory with given alignment and number of elements
     */
    template <size_t Alignment>
    static inline pointer aligned_alloc(size_t count) noexcept
    {
        return static_cast<pointer>(std::aligned_alloc(Alignment, get_bytesize(count)));
    }

    /**
     * @brief Allocate aligned memory with given number of lemenets
     */
    static inline pointer aligned_alloc(size_t count) noexcept
    {
        return aligned_alloc<alignment>(count);
    }

    static inline void aligned_free(pointer src) noexcept
    {
        std::free(src);
    }

    // Default c'tor
    tensor() : 
        m_shape(), m_tensor(internal_data{0}), 
        m_data_type(data_type::buffer), m_bytesize(0),
        m_borrowed(false) {}


   /**
    * @brief Create a tensor object
    * @param dim dimensions of the tensor
    * @param init initial value for each 'cell' in a tensor
    */
    tensor(const shape& dim, number_t init = 0.0) noexcept
    : m_shape(dim)
    {
        auto count      = dim.total();
        m_bytesize      = get_bytesize(count);
        m_tensor.data   = aligned_alloc(count);
        m_data_type     = data_type::buffer;
        m_borrowed      = false;

        if (init != 0.0)
            std::fill_n(m_tensor.data, count, init);
    }

    // Copy constructor
    tensor(const tensor& t) noexcept { void(*this = t); }

    // Move constructor
    tensor(tensor&& t) noexcept { void(*this = std::forward<tensor>(t)); }

    // Takes the ownership of the `data`, might blow your leg
    tensor(pointer data, const shape& dim) noexcept : m_shape(dim)
    {
        m_tensor.data   = data;
        m_data_type     = data_type::buffer;
        m_bytesize      = get_bytesize(m_shape.total());
        m_borrowed      = false; // the 'borrowed' buffer is not ours
    }

    tensor& operator=(const tensor& t) noexcept
    {
        auto count = M_alloc_buffer(t);

        if (count == 0)
            return *this;

        // If that's a cuda pointer, memcpy to this buffer
        if (t.m_data_type == data_type::cuda && t.m_tensor.cu_ptr != 0) {
            cuMemcpyDtoH(
                m_tensor.data, t.m_tensor.cu_ptr, 
                count * sizeof(number_t));
        }
        else
        {
            std::copy_n(t.m_tensor.data, count, m_tensor.data); // will memcpy the buffer
        }

        return *this;
    }

    tensor& operator=(tensor&& t) noexcept
    {
        m_bytesize  = t.m_bytesize;
        m_shape     = std::forward<shape>(t.m_shape);
        m_data_type = data_type::buffer;
        m_borrowed  = false; // since I'm getting the ownership of the pointer

        // If that's a cuda pointer, copy the buffer
        if (t.m_data_type == data_type::cuda && t.m_tensor.cu_ptr != 0)
        {
            const auto count    = size();
            m_tensor.data       = aligned_alloc(count);
            cuMemcpyDtoH(
                m_tensor.data, t.m_tensor.cu_ptr, 
                count * sizeof(number_t));
        }
        else
        {
            // Just release the buffer
            m_tensor.data = t.release();
        }

        return *this;
    }

    virtual ~tensor() noexcept
    {
        if (m_borrowed)
            return;
        
        if (m_data_type == data_type::buffer) {
            aligned_free(m_tensor.data);
            m_tensor.data = nullptr;
        }
    }

    // Get the dimensions (as a vector)
    shape::dim_t dim() const noexcept { return m_shape.dim(); }
    const shape& layout() const noexcept { return m_shape; }

    // Get the rank of the tensor
    size_t rank() const noexcept { return m_shape.rank(); }

    // Get total size of the internall buffer
    size_t size() const noexcept { return m_shape.total(); }

    // Get number of bytes memory holds (doesn't have to be sames as size*sizeof(number_t) since it's aligned)
    size_t bytesize() const noexcept { return m_bytesize; }

    // Get buffer type
    data_type type() const noexcept { return m_data_type; }

    // Check wheter internal buffer is stored in gpu memory
    bool is_cuda() const noexcept { return m_data_type == data_type::cuda; }

    // Check if the tensor is empty
    bool empty() const noexcept 
    { 
        return m_shape.m_dims.empty() || (
            m_data_type == data_type::buffer ? m_tensor.data == nullptr : m_tensor.cu_ptr == 0
        ); 
    }

    // Get the internal 1d buffer
    pointer data() noexcept { return m_tensor.data; }
    const_pointer data() const noexcept { return m_tensor.data; }

    // Release the buffer, should be wrapped in a unique pointer with array type
    pointer release() noexcept { return M_release_t<pointer>(); }
    
    // Print the tensor to output stream
    friend std::ostream& operator<<(std::ostream& out, const tensor& t) noexcept
    {
        out << "<tensor: dtype=" << utils::TypeName<number_t>() << " " << t.m_shape << ">\n";

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

    // Private constructor for optimized cuda buffer management
    tensor(cu_pointer cu_ptr, shape dim) noexcept : m_shape(dim) 
    {
        m_tensor.cu_ptr = cu_ptr;
        m_data_type     = data_type::cuda;
        m_bytesize      = get_bytesize(m_shape.total());
    }

    cu_pointer cu_release() noexcept { return M_release_t<cu_pointer>(); }
    cu_pointer cu_data() const noexcept { return m_tensor.cu_ptr; }


    shape m_shape;
    internal_data m_tensor;
    data_type m_data_type;
    size_t m_bytesize;
    bool m_borrowed;

    /**
     * @brief Allocates the buffer with the same size as `t`, copies the dimension, and sets the 
     * data type to buffer, but DOES NOT COPY THE CONTENT of t's data
     * @returns t.size() (if 0 then buffer was not allocated and is set to `nullptr`)
     */
    inline size_t M_alloc_buffer(const tensor& t) noexcept
    {
        const auto count    = t.size();
        m_shape             = t.m_shape;
        m_borrowed          = false;
        m_data_type         = data_type::buffer; // always use buffer
        m_tensor.data       = nullptr;
        m_bytesize          = 0;
        
        if (count == 0) 
            return 0;

        m_bytesize          = get_bytesize(count);
        m_tensor.data       = aligned_alloc(count);
        return count;
    }

    // Print the tensor recursively, to given output stream, rank = t.rank(), index = 0, offset = 0
    static void M_print_tensor(const tensor& t, std::ostream& out, size_t rank, size_t index = 0, size_t offset = 0) noexcept
    {
        auto end = t.m_shape.m_dims.at(index);

        // Text tabluation
        for (size_t p = 0; p < index; p++)
            out << ' ';
        
        // Got 1D representation
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
                // Go to next dimension of the tensor
                M_print_tensor(t, out, rank - 1, index + 1, offset + i * t.m_shape.m_dims[index + 1]);
            }

            // Text tabluation
            for (size_t p = 0; p < index; p++)
                out << ' ';
        }


        if (rank != t.rank())
            out << "],\n";
        else
            out << "]\n";
    }


    // Get the internal buffer, either as a `pointer` or `cu_pointer`
    template <typename T>
    inline std::enable_if_t<std::is_same_v<T, pointer> || std::is_same_v<T, cu_pointer>, T>
    M_release_t() noexcept
    {
        T res;
        if constexpr (std::is_same_v<T, pointer>) { 
            res = m_tensor.data; 
            m_tensor.data = nullptr; 
        }
        else { 
            res = m_tensor.cu_ptr; 
            m_tensor.cu_ptr = 0; 
        }
        m_bytesize = 0;
        m_shape.clear();
        return res;
    }
};


END_BLUST_NAMESPACE

#endif //TENSOR_HPP
