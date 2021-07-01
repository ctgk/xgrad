#include <algorithm>
#include <memory>
#include <string>

#include "xgrad/core/ndarray.hpp"

namespace xgrad
{

static inline std::size_t shape_to_size(const std::vector<std::size_t>& shape)
{
    if (shape.size() == 0UL) {
        return 1UL;
    }
    std::size_t size = 1UL;
    for (auto&& length : shape) {
        size *= length;
    }
    return size;
}

static inline std::vector<std::size_t>
shape_to_strides(const std::vector<std::size_t>& shape)
{
    if (shape.size() == 0UL) {
        return std::vector<std::size_t>();
    }
    auto strides = std::vector<std::size_t>(shape.size(), 1UL);
    for (auto i = shape.size() - 1UL; i--;) {
        strides[i] = strides[i + 1UL] * shape[i + 1UL];
    }
    return strides;
}

template <class T>
ndarray<T>::ndarray() : ndarray<T>(std::vector<std::size_t>(), nullptr)
{
}

static inline std::size_t
index_of_last_element(const std::vector<std::size_t>& shape)
{
    auto index = 1UL;
    for (auto&& length : shape) {
        index *= length;
    }
    index -= 1UL;
    return index;
}

static inline bool
in(const std::vector<std::size_t>& vec, const std::size_t value)
{
    return std::find(vec.cbegin(), vec.cend(), value) != vec.cend();
}

template <class T>
ndarray<T>::ndarray(
    const std::vector<std::size_t>& shape,
    const std::shared_ptr<std::vector<T>>& data)
    : m_ndim(shape.size()), m_shape(shape), m_strides(shape_to_strides(shape)),
      m_size(shape_to_size(shape)),
      m_data(
          (data == nullptr) ? std::make_shared<std::vector<T>>(m_size) : data)
{
    if (in(shape, 0UL)) {
        throw std::invalid_argument("0 in `shape`");
    }
    const auto size = index_of_last_element(shape) + 1UL;
    if (m_data->size() != size) {
        throw std::invalid_argument(
            "Size of data vector should be " + std::to_string(size) + ", not "
            + std::to_string(m_data->size()));
    }
}

static inline void throw_error_if_axis_larger_than_ndim(
    const std::size_t axis, const std::size_t ndim)
{
    if (axis >= ndim) {
        throw std::invalid_argument(
            "`axis` ( " + std::to_string(axis)
            + ") must be smaller than the dimensionality ("
            + std::to_string(ndim) + ")");
    }
}

template <class T>
std::size_t ndarray<T>::ndim() const
{
    return m_ndim;
}

template <class T>
const std::vector<std::size_t>& ndarray<T>::shape() const
{
    return m_shape;
}

template <class T>
std::size_t ndarray<T>::shape(const std::size_t axis) const
{
    throw_error_if_axis_larger_than_ndim(axis, m_ndim);
    return m_shape[axis];
}

template <class T>
const std::vector<std::size_t>& ndarray<T>::strides() const
{
    return m_strides;
}

template <class T>
std::size_t ndarray<T>::strides(const std::size_t axis) const
{
    throw_error_if_axis_larger_than_ndim(axis, m_ndim);
    return m_strides[axis];
}

template class ndarray<float>;
template class ndarray<double>;

} // namespace xgrad
