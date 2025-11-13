#pragma once

#include <functional>
#include <memory>
#include <type_traits>

#include "base_types.hpp"

START_BLUST_NAMESPACE

template <typename dtype>
concept IsDType = std::is_floating_point_v<dtype> || std::is_same_v<dtype, CUdeviceptr>;


template <IsDType dtype>
class internal_tensor_data {
public:
    typedef CUdeviceptr cu_pointer;
    typedef CUdeviceptr& cu_pointer_ref;
    typedef dtype* pointer;
    typedef const dtype* const_pointer;
    typedef std::function<dtype()> gen_fn;

    // Memory alignment
    static constexpr size_t alignment = 32;

    // bytesize getter
    inline size_t get_bytesize() const noexcept {
        return m_bytesize;
    }

    // get total count (may not be equal to bytesize/sizeof(dtype))
    inline size_t size() const noexcept {
        return m_size;
    }

    // void build(size_t count, dtype init) = 0;
    virtual void fill(dtype v) noexcept = 0;
    virtual void generate(gen_fn gen) noexcept = 0;
protected:
    size_t m_bytesize{0};
    size_t m_size{0};
};

template <IsDType dtype>
class tensor_buffer : public internal_tensor_data<dtype> {
    std::unique_ptr<dtype> m_data{nullptr};
public:
    using typename internal_tensor_data<dtype>::pointer;
    using typename internal_tensor_data<dtype>::const_pointer;
    using typename internal_tensor_data<dtype>::gen_fn;
    static constexpr auto alignment = internal_tensor_data<dtype>::alignment;

    // Construct new buffer
    tensor_buffer(size_t count, dtype init) {
        this->m_size = count;
        this->m_data.reset(utils::aligned_alloc<alignment, dtype>(count));
        this->m_bytesize = utils::get_bytesize<alignment, dtype>(count);
        std::fill_n(this->m_data.get(), count, init);
    }

    // Claim given pointer
    explicit
    tensor_buffer(pointer data, size_t count) {
        this->m_size = count;
        this->m_bytesize = utils::get_bytesize<alignment, dtype>(count);
        this->m_data.reset(data);
    }

    // Copy all data from given pointer
    explicit
    tensor_buffer(const_pointer data, size_t count) {
        this->m_size = count;
        this->m_bytesize = utils::get_bytesize<alignment, dtype>(count);
        this->m_data.reset(utils::aligned_alloc<alignment, dtype>(count));
        std::copy_n(data, this->m_size, m_data.get());
    }

    tensor_buffer* clone() {
        return new tensor_buffer(const_pointer(this->m_data.get()), this->m_size);
    }

    void fill(dtype init) noexcept {
        std::fill_n(m_data.get(), this->m_size, init);
    }

    void generate(gen_fn gen) noexcept {
        std::generate_n(m_data.get(), this->m_size, gen);
    }

    pointer begin() const noexcept {
        return m_data.get();
    }

    pointer end() const noexcept {
        return m_data.get() + this->m_size;
    }

    pointer data() noexcept {
        return m_data.get();
    }

    const_pointer data() const noexcept {
        return m_data.get();
    }

    pointer release() noexcept {
        return m_data.release();
    }
};

template <IsDType dtype>
class tensor_cuda_buffer : public internal_tensor_data<dtype> {
public:
    using typename internal_tensor_data<dtype>::cu_pointer;
    using typename internal_tensor_data<dtype>::gen_fn;

    // Claim pointer
    explicit
    tensor_cuda_buffer(cu_pointer data, size_t count) {
        ptr = data;
        this->m_bytesize = this->m_size = count;
    }

    tensor_cuda_buffer(size_t count, dtype init) {
        // Create the data
    }

    ~tensor_cuda_buffer() {
        // Clean up the buffer
    }

    tensor_cuda_buffer* clone() {
        return new tensor_cuda_buffer(size_t(0), 0);
    }

    void fill(dtype init) noexcept {
        // std::fill_n(m_data.get(), m_size, init);
    }

    void generate(gen_fn gen) noexcept {
        // std::generate_n(m_data.get(), m_size, gen);
    }

    cu_pointer data() const noexcept {
        return ptr;
    }

    cu_pointer release() noexcept {
        auto ret = ptr;
        ptr = 0;
        return ret;
    }

private:
    cu_pointer ptr{0};
};

END_BLUST_NAMESPACE