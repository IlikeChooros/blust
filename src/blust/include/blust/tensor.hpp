#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <vector>
#include <algorithm>
#include <numeric>
#include <memory>
#include <variant>
#include <functional>

#include "utils.hpp"
#include "shape.hpp"
#include "data_handler.hpp"

START_BLUST_NAMESPACE

template <typename T>
concept IsPointerOrCU = std::is_same_v<T, number_t*> || std::is_same_v<T, CUdeviceptr>;

template <typename T>
concept TensorDataType = std::is_same_v<T, float> || std::is_same_v<T, double> || std::is_integral_v<T>;


// Main tensor class, can either hold heap memory buffer, or gpu memory pointer
// The buffer is 32-byte aligned
class tensor 
{
public:
    friend class operations;
    friend class cpu_ops;
    friend class ops_tensor;

    typedef CUdeviceptr cu_pointer;
    typedef CUdeviceptr& cu_pointer_ref;
    typedef number_t* pointer;
    typedef const number_t* const_pointer;

    typedef std::variant<cu_pointer, pointer> internal_data; 
    typedef data_handler<number_t>::pointer_type pointer_type;
    

    static int n_allocs;
    static int max_allocs;

    static void inc_alloc(int n = 1) {
        max_allocs = std::max(max_allocs, n_allocs += n); 
    }

    // Memory alignment
    static constexpr size_t alignment = 32;

    /**
     * @brief Get total size in bytes, with given alignment
     */
    template <size_t Alignment, typename dtype>
    static constexpr size_t get_bytesize(size_t count) noexcept
    {
        return ((count * sizeof(dtype) + Alignment - 1) / Alignment ) * Alignment;
    }

    /**
     * @brief Get total size in bytes
     */
    static constexpr size_t get_bytesize(size_t count) noexcept
    {
        return get_bytesize<alignment, number_t>(count);
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

    // Free the aligned memory
    static inline void aligned_free(pointer src) noexcept
    {
        std::free(src);
    }

    // Default c'tor
    tensor() = default;

   /**
    * @brief Create a tensor object
    * @param dim dimensions of the tensor
    * @param init initial value for each 'cell' in a tensor
    */
    tensor(const shape& dim, number_t init = 0.0) noexcept
    : m_shape(dim)
    {
        inc_alloc(1);
        // auto count      = dim.total();
        // m_bytesize      = get_bytesize(count);
        // m_data_type     = pointer_type::host;
        // m_tensor        = aligned_alloc(count);

        m_handler = data_handler<number_t>(dim, init, pointer_type::host);

        // if (init != 0.0)
        // std::fill_n(std::get<pointer>(m_tensor), count, init);
    }

    // Copy constructor
    tensor(const tensor& t) noexcept { void(*this = t); }

    // Move constructor
    tensor(tensor&& t) noexcept { void(*this = std::forward<tensor>(t)); }

    // Takes the ownership of the `data`, might blow your leg
    tensor(pointer data, const shape& dim) noexcept : m_shape(dim)
    {
        m_handler.build(data, dim);
    }

    // Copy dim.total() elements from the given data to the internal buffer
    tensor(const_pointer data, const shape& dim) noexcept : m_shape(dim)
    {
        // copy the data to the internal buffer
        inc_alloc(1);
        m_handler.build(data, dim);
        // auto count                  = m_shape.total();
        // m_bytesize                  = get_bytesize(m_shape.total());
        // m_tensor                    = aligned_alloc(m_bytesize);
        // std::copy_n(data, m_shape.total(), std::get<pointer>(m_tensor));
        // m_data_type                 = pointer_type::host;
    }

    tensor& operator=(const tensor& t) noexcept {
        m_handler = t.m_handler;
        m_shape = t.m_shape;
        return *this;
    }

    tensor& operator=(tensor&& t) noexcept {
        m_handler = std::move(t.m_handler);
        m_shape = std::move(t.m_shape);
        return *this;
    }

    virtual ~tensor() noexcept { M_cleanup_buffer(); }

    // Allocate underlying tensor buffer filled with given `init` value
    void build(const shape& dim, number_t init = 0.0) noexcept 
    {
        inc_alloc(1);
        m_handler.build(dim, init, pointer_type::host);
        // auto count      = dim.total();
        // m_bytesize      = get_bytesize(count);
        // m_data_type     = pointer_type::host;
        // m_tensor        = aligned_alloc(count);

        // if (init != 0.0)
        //     std::fill_n(std::get<pointer>(m_tensor), count, init);
    }

    // Get the dimensions (as a vector)
    shape::dim_t dim() const noexcept { return m_shape.dim(); }
    const shape& layout() const noexcept { return m_shape; }

    // Get the rank of the tensor
    size_t rank() const noexcept { return m_shape.rank(); }

    // Get total size of the internall buffer
    size_t size() const noexcept { return m_handler.size(); }

    // Get number of bytes memory holds (doesn't have to be sames as size*sizeof(number_t) since it's aligned)
    size_t bytesize() const noexcept { return m_handler.bytesize(); }

    // Get buffer type
    pointer_type type() const noexcept { return m_handler.type(); }

    // Check wheter internal buffer is stored in gpu memory
    bool is_cuda() const noexcept { return m_handler.is_cuda(); }

    // fill the tensor with given predicate
    void fill(std::function<number_t()> f) noexcept {
        m_handler.generate(f);
    }

    // fill the tensor with given value
    void fill(number_t val) noexcept {
        m_handler.fill(val);
    }

    // Check if the tensor is empty
    bool empty() const noexcept 
    {
        return m_handler.empty();
    }

    // Get the internal 1d buffer
    pointer data() noexcept { return m_handler.data();}
    const_pointer data() const noexcept { return m_handler.data();}

    pointer begin() noexcept { return m_handler.begin(); }
    auto begin() const noexcept { return m_handler.begin(); }
    pointer end() noexcept { return m_handler.end(); }
    auto end() const noexcept { return m_handler.end(); }

    // Release the buffer, should be wrapped in a unique pointer with array type
    pointer release() noexcept { return m_handler.release(); }
    
    // Print the tensor to output stream
    friend std::ostream& operator<<(std::ostream& out, const tensor& t) noexcept;

    // Array like access
    number_t& operator()(size_t i) { return *(data() + i); }
    number_t operator()(size_t i) const { return *(data() + i); }

private:

    // Private constructor for optimized cuda buffer management
    tensor(cu_pointer cu_ptr, shape dim) noexcept : m_shape(dim) 
    {
        m_handler.build(cu_ptr, dim);
        // std::get<cu_pointer>(m_tensor)  = cu_ptr;
        // m_data_type                     = pointer_type::device;
        // m_bytesize                      = get_bytesize(m_shape.total());
    }

    cu_pointer cu_release() noexcept { return m_handler.cu_release(); }
    cu_pointer cu_data() const noexcept { return m_handler.cu_data(); }
    // cu_pointer_ref cu_data() noexcept { return m_handler.cu_data(); }

    shape m_shape{};
    // internal_data m_tensor{};
    // pointer_type m_data_type{pointer_type::host};
    // size_t m_bytesize{};
    // bool m_shared{false};

    data_handler<number_t> m_handler;


    // inline size_t M_alloc_buffer(const tensor& t) noexcept {}
    
    /**
     * @brief Borrow the buffer from given tensor, will share the buffer with 't'
     */
    inline void M_borrow(const tensor& t) noexcept
    {
        m_shape   = t.m_shape;
        m_handler = t.m_handler;
        // m_tensor    = t.m_tensor;
        // m_data_type = t.m_data_type;
        // m_bytesize  = t.m_bytesize;
        // m_shared    = true;
    }

    // Delete the buffer if not shared
    inline void M_cleanup_buffer() noexcept
    {
        // if (m_shared)
        //     return;
    
        // if (
        //     m_data_type == pointer_type::host 
        //     && std::holds_alternative<pointer>(m_tensor) 
        //     && data() != nullptr) 
        // {
        //     aligned_free(data());
        //     data() = nullptr;
        //     inc_alloc(-1);
        // }
    }

    // Print the tensor recursively, to given output stream, rank = t.rank(), index = 0, offset = 0
    static void M_print_tensor(
        const tensor& t, std::ostream& out, size_t rank, 
        size_t index = 0, size_t offset = 0
    ) noexcept;

    // Get the internal buffer, either as a `pointer` or `cu_pointer`
    template <IsPointerOrCU T>
    inline T M_release_t() noexcept
    {

        // T res       = std::get<T>(m_tensor);
        // m_tensor    = {};
        // m_bytesize  = 0;
        // m_shape.clear();
        // return res;
    }
};

END_BLUST_NAMESPACE

#endif //TENSOR_HPP
