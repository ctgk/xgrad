#include <memory>

#include "xgrad/core/ndarray.hpp"
#include "xgrad/core/ndshape.hpp"
#include "xgrad/core/node.hpp"

namespace xgrad
{

namespace internal
{

template <class T>
const std::shared_ptr<ndarray_node<T>>& get_node(const ndarray<T>& a)
{
    return a.m_node;
}

template const std::shared_ptr<ndarray_node<float>>&
get_node<float>(const ndarray<float>&);
template const std::shared_ptr<ndarray_node<double>>&
get_node<double>(const ndarray<double>&);

template <class T>
ndarray<T> create_ndarray(const std::shared_ptr<operation_node<T>>& op)
{
    return ndarray<T>(op);
}

template ndarray<float>
create_ndarray(const std::shared_ptr<operation_node<float>>& op);
template ndarray<double>
create_ndarray(const std::shared_ptr<operation_node<double>>& op);

} // namespace internal

template <class T>
ndarray<T>::ndarray(const std::shared_ptr<internal::operation_node<T>>& op)
    : m_node(std::make_shared<internal::ndarray_node<T>>(op))
{
}

template <class T>
ndarray<T>::ndarray() : ndarray<T>(ndshape(), nullptr)
{
}

template <class T>
ndarray<T>::ndarray(
    const ndshape& shape, const std::shared_ptr<std::vector<T>>& data)
    : m_node(std::make_shared<internal::ndarray_node<T>>(shape, data))
{
    const auto data_size = m_node->data()->size();
    if (data_size != shape.product()) {
        throw std::invalid_argument(
            "Size of data vector should be " + std::to_string(shape.product())
            + ", not " + std::to_string(data_size));
    }
}

template <class T>
ndarray<T>::ndarray(const ndshape& shape, const std::vector<T>& data)
    : ndarray(shape, std::make_shared<std::vector<T>>(data))
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
bool ndarray<T>::is_view() const
{
    return m_node->is_view();
}

template <class T>
std::size_t ndarray<T>::ndim() const
{
    return m_node->shape().ndim();
}

template <class T>
std::size_t ndarray<T>::size() const
{
    return m_node->shape().product();
}

template <class T>
const ndshape& ndarray<T>::shape() const
{
    return m_node->shape();
}

template <class T>
std::size_t ndarray<T>::shape(const std::size_t axis) const
{
    return m_node->shape()[axis];
}

template <class T>
const std::vector<std::size_t>& ndarray<T>::strides() const
{
    return m_node->strides();
}

template <class T>
std::size_t ndarray<T>::strides(const std::size_t axis) const
{
    throw_error_if_axis_larger_than_ndim(axis, ndim());
    return m_node->strides()[axis];
}

template <class T>
T* ndarray<T>::data()
{
    return m_node->data()->data();
}

template <class T>
const T* ndarray<T>::cdata() const
{
    return m_node->data()->data();
}

template <class T>
T* ndarray<T>::grad()
{
    return m_node->grad()->data();
}

template <class T>
const T* ndarray<T>::cgrad() const
{
    return m_node->grad()->data();
}

template <class T>
bool ndarray<T>::requires_grad() const
{
    return m_node->requires_grad();
}

template <class T>
ndarray<T> ndarray<T>::grad_to_array() const
{
    if (is_view()) {
        throw std::runtime_error(
            "grad_to_array is not implemented yet for a view of an array");
    }
    return ndarray(shape(), *m_node->grad());
}

template <class T>
void ndarray<T>::requires_grad(const bool flag)
{
    return m_node->requires_grad(flag);
}

template <class T>
void ndarray<T>::backward()
{
    m_node->backward();
}

template <class T>
std::size_t ndarray<T>::num_backward() const
{
    return m_node->num_backward();
}

template class ndarray<float>;
template class ndarray<double>;

} // namespace xgrad
