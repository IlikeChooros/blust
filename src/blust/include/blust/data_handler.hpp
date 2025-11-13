#pragma once

#include <variant>
#include <memory>
#include <type_traits>

#include "base_types.hpp"
#include "internal_tensor_data.hpp"

START_BLUST_NAMESPACE

template <IsDType dtype>
class data_handler {
public:
    typedef enum class pointer_type { host = 1, device = 2 } pointer_type;
    typedef CUdeviceptr cu_pointer;
    typedef CUdeviceptr& cu_pointer_ref;
    typedef dtype* pointer;
    typedef const dtype* const_pointer;
    typedef std::shared_ptr<tensor_buffer<dtype>> shared_buffer_ptr;
    typedef std::shared_ptr<tensor_cuda_buffer<dtype>> shared_cu_ptr;
    typedef std::variant<shared_buffer_ptr, shared_cu_ptr> variant_data;

    data_handler() = default;

    data_handler(shape dim, dtype init, pointer_type type) {
        build(std::forward<shape>(dim), init, type);
    }

    data_handler(const data_handler<dtype>& other) {
        void(*this = other);
    }

    data_handler(data_handler<dtype>&& other) {
        void(*this = std::forward<data_handler<dtype>>(other));
    }

    // Copies all conent of the `other` allocated memory
    data_handler<dtype>& operator=(const data_handler<dtype>& other) {
        // m_data = other.m_data;
        m_type = other.m_type;
        if (m_type == pointer_type::host) {
            m_data = shared_buffer_ptr(
                std::get<shared_buffer_ptr>(other.m_data)->clone());
        } else {
            m_data = shared_cu_ptr(
                std::get<shared_cu_ptr>(other.m_data)->clone());
        }
        M_set_base_ptr();
        return *this;
    }

    // Moves all other's conent into this object
    data_handler<dtype>& operator=(data_handler<dtype>&& other) {
        m_data = std::move(other.m_data);
        m_type = other.m_type;
        M_set_base_ptr();
        return *this;
    }

    /**
     * Take ownership of the given buffer pointer
     */
    inline void build(pointer data, shape dim) noexcept {
        m_type = pointer_type::host;
        m_data = shared_buffer_ptr(new tensor_buffer<dtype>(data, dim.total()));
        M_set_base_ptr();
    }

    /**
     * Deep copy the `data` contents
     */
    inline void build(const_pointer data, shape dim) noexcept {
        m_type = pointer_type::host;
        m_data = shared_buffer_ptr(new tensor_buffer<dtype>(data, dim.total()));
        M_set_base_ptr();
    }

    inline void build(cu_pointer data, shape dim) noexcept {
        m_type = pointer_type::device;
        m_data = shared_cu_ptr(new tensor_cuda_buffer<dtype>(data, dim.total()));
        M_set_base_ptr();
    }

    /**
     * @brief Allocate internal buffer given dims and initial value
     */
    inline void build(shape dim, dtype init, pointer_type type) noexcept {
        m_type = type;
        if (type == pointer_type::device) {
            m_data = shared_cu_ptr(
                new tensor_cuda_buffer<dtype>(std::move(dim.total()), init));
        } else {
            m_data = shared_buffer_ptr(
                new tensor_buffer<dtype>(std::move(dim.total()), init));
        }
        M_set_base_ptr();
    }

    /**
     * @brief Make sure the underlying buffer is not shared 
     * (the atomic count of shared ptr == 1)
     */
    inline void ensure_unique() noexcept {
        if (m_type == pointer_type::host) {
            auto& sp = std::get<shared_buffer_ptr>(m_data);

            if (!sp || sp.use_count() == 1) {
                return;
            }

            m_data = shared_buffer_ptr(sp->clone());
            // auto clone = std::make_shared<tensor_buffer<dtype>>(sp->size(), dtype{});
            // std::copy_n(sp->data(), sp->size(), clone->data());
            // sp = std::move(clone);
        } else {
            auto& sp = std::get<shared_cu_ptr>(m_data);
            
            if (!sp || sp.use_count() == 1) {
                return;
            }
            
            m_data = shared_cu_ptr(sp->clone());
            // clone CUDA buffer (device-to-device copy)
            // auto clone = std::make_shared<tensor_cuda_buffer<dtype>>(sp->size(), dtype{});
            // cudaMemcpy(clone->data(), sp->data(), bytes, cudaMemcpyDeviceToDevice);
            // sp = std::move(clone);
        }
        M_set_base_ptr();
    }

    /**
     * @brief Fill internal buffer with given value
     */
    inline void fill(dtype init) noexcept {
        m_base_ptr->fill(init);
    }

    /**
     * @brief Apply given generator to each element of the tensor
     */
    inline void generate(std::function<dtype()> gen) noexcept {
        m_base_ptr->generate(gen);
    }

    inline bool empty() const noexcept {
        return m_base_ptr == nullptr || m_base_ptr->size() == 0;
    }

    inline size_t size() const noexcept {
        return m_base_ptr->size();
    }

    inline size_t bytesize() const noexcept {
        return m_base_ptr->get_bytesize();
    }

    inline bool is_cuda() const noexcept {
        return m_type == pointer_type::device;
    }

    pointer_type type() const noexcept {
        return m_type;
    }

    cu_pointer cu_data() const noexcept {
        return std::get<shared_cu_ptr>(m_data)->data();
    }

    const const_pointer data() const noexcept {
        return std::get<shared_buffer_ptr>(m_data)->data();
    }

    pointer data() noexcept {
        return std::get<shared_buffer_ptr>(m_data)->data();
    }

    pointer begin() const noexcept {
        return std::get<shared_buffer_ptr>(m_data)->begin();
    }

    pointer end() const noexcept {
        return std::get<shared_buffer_ptr>(m_data)->end();
    }

    pointer release() noexcept {
        return std::get<shared_buffer_ptr>(m_data)->release();
    }

    cu_pointer cu_release() noexcept {
        return std::get<shared_cu_ptr>(m_data)->release();
    }

private:


    void M_set_base_ptr() noexcept {
        if (std::holds_alternative<shared_buffer_ptr>(this->m_data)) {
            this->m_base_ptr = std::get<shared_buffer_ptr>(this->m_data).get();
        } else {
            this->m_base_ptr = std::get<shared_cu_ptr>(this->m_data).get();
        }
    }

    variant_data m_data{shared_buffer_ptr{nullptr}};
    pointer_type m_type{pointer_type::host};
    internal_tensor_data<dtype>* m_base_ptr{nullptr};
};

END_BLUST_NAMESPACE