#pragma once

#include "types.hpp"


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

    static int n_allocs;
    static int max_allocs;

    static void inc_alloc(int n = 1) {
        max_allocs = std::max(max_allocs, n_allocs += n); 
    }

    // Memory alignment
    static constexpr size_t alignment = 32;

    // bytesize getter
protected:
    size_t m_bytesize;
};


template <IsDType dtype>
class tensor_buffer : public internal_tensor_data<dtype> {
    
    std::unique_ptr<dtype> m_data;
public:
    typedef dtype* pointer;

    heap_data(heap_data&& data) {
        m_data = std::move(data.m_data);
    }

    heap_data(size_t count, dtype init) {
        build(std::move(count), init);
    }

    void build(size_t count, dtype init) {
        m_data = utils::aligned_alloc<alignment, dtype>(count);
        std::fill_n(m_data.get(), count, init);
    }

    // fill
};

template <IsDType dtype>
class tensor_cuda_buffer : internal_tensor_data<dtype> {
public:
    typedef CUdeviceptr cu_pointer;

    cu_data(shape dim, dtype init) {
        // Create the data
    }

    cu_data(const cu_data& data) {
        // Make a full copy of the data
    }

    cu_data(cu_data&& data) {
        // Take ownership of the data
    }

private:
    cu_pointer ptr;
};


END_BLUST_NAMESPACE