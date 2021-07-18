#include <memory>

#include "xgrad/core/ndshape.hpp"
#include "xgrad/core/node.hpp"
#include "xgrad/core/tensor.hpp"

namespace xgrad
{

namespace internal
{

template <class T>
const std::shared_ptr<tensor_node<T>>& get_node(const tensor<T>& a)
{
    return a.m_node;
}

template const std::shared_ptr<tensor_node<float>>&
get_node<float>(const tensor<float>&);
template const std::shared_ptr<tensor_node<double>>&
get_node<double>(const tensor<double>&);

template <class T>
tensor<T> create_tensor(const std::shared_ptr<operation_node<T>>& op)
{
    return tensor<T>(op);
}

template tensor<float>
create_tensor(const std::shared_ptr<operation_node<float>>& op);
template tensor<double>
create_tensor(const std::shared_ptr<operation_node<double>>& op);

} // namespace internal

template <class T>
tensor<T>::tensor(const std::shared_ptr<internal::operation_node<T>>& op)
    : m_node(std::make_shared<internal::tensor_node<T>>(op))
{
}

template <class T>
tensor<T>::tensor() : tensor<T>(ndshape(), nullptr)
{
}

template <class T>
tensor<T>::tensor(
    const ndshape& shape, const std::shared_ptr<std::vector<T>>& data)
    : m_node(std::make_shared<internal::tensor_node<T>>(shape, data))
{
    const auto data_size = m_node->data()->size();
    if (data_size != shape.product()) {
        throw std::invalid_argument(
            "Size of data vector should be " + std::to_string(shape.product())
            + ", not " + std::to_string(data_size));
    }
}

template <class T>
tensor<T>::tensor(const ndshape& shape, const std::vector<T>& data)
    : tensor(shape, std::make_shared<std::vector<T>>(data))
{
}

static inline void throw_error_if_axis_larger_than_ndim(
    const std::size_t axis, const std::size_t ndim)
{
    if (axis >= ndim) {
        throw std::out_of_range(
            "`axis` ( " + std::to_string(axis)
            + ") must be smaller than the dimensionality ("
            + std::to_string(ndim) + ")");
    }
}

template <class T>
bool tensor<T>::is_view() const
{
    return m_node->is_view();
}

template <class T>
std::size_t tensor<T>::ndim() const
{
    return m_node->shape().ndim();
}

template <class T>
std::size_t tensor<T>::size() const
{
    return m_node->shape().product();
}

template <class T>
const ndshape& tensor<T>::shape() const
{
    return m_node->shape();
}

template <class T>
std::size_t tensor<T>::shape(const std::size_t axis) const
{
    return m_node->shape()[axis];
}

template <class T>
const std::vector<std::size_t>& tensor<T>::strides() const
{
    return m_node->strides();
}

template <class T>
std::size_t tensor<T>::strides(const std::size_t axis) const
{
    throw_error_if_axis_larger_than_ndim(axis, ndim());
    return m_node->strides()[axis];
}

template <class T>
T* tensor<T>::data()
{
    return m_node->data()->data();
}

template <class T>
const T* tensor<T>::cdata() const
{
    return m_node->data()->data();
}

template <class T>
T* tensor<T>::grad()
{
    return m_node->grad()->data();
}

template <class T>
const T* tensor<T>::cgrad() const
{
    return m_node->grad()->data();
}

template <class T>
bool tensor<T>::requires_grad() const
{
    return m_node->requires_grad();
}

template <class T>
tensor<T> tensor<T>::grad_to_array() const
{
    if (is_view()) {
        throw std::runtime_error(
            "grad_to_array is not implemented yet for a view of an array");
    }
    return tensor(shape(), *m_node->grad());
}

template <class T>
void tensor<T>::requires_grad(const bool flag)
{
    return m_node->requires_grad(flag);
}

template <class T>
void tensor<T>::backward()
{
    m_node->backward();
}

template <class T>
std::size_t tensor<T>::num_backward() const
{
    return m_node->num_backward();
}

template class tensor<float>;
template class tensor<double>;

} // namespace xgrad
