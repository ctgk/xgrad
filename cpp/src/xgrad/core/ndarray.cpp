#include <memory>

#include "xgrad/core/ndarray.hpp"
#include "xgrad/core/ndshape.hpp"
#include "xgrad/core/node.hpp"

namespace xgrad
{

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
std::size_t ndarray<T>::ndim() const
{
    return m_node->shape().ndim();
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

template class ndarray<float>;
template class ndarray<double>;

} // namespace xgrad
