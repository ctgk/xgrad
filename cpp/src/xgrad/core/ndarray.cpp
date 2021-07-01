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
ndarray<T>::ndarray() : ndarray<T>(nullptr, std::vector<std::size_t>())
{
}

template <class T>
ndarray<T>::ndarray(
    const std::shared_ptr<std::vector<T>>& data,
    const std::vector<std::size_t>& shape)
    : ndarray<T>(data, shape, shape_to_strides(shape), 0UL)
{
}

static inline std::size_t index_of_last_element(
    const std::vector<std::size_t>& shape,
    const std::vector<std::size_t>& strides,
    const std::size_t offset)
{
    auto index = offset;
    for (auto n = shape.size(); n--;) {
        index += (shape[n] - 1UL) * strides[n];
    }
    return index;
}

static inline bool
in(const std::vector<std::size_t>& vec, const std::size_t value)
{
    return std::find(vec.cbegin(), vec.cend(), value) != vec.cend();
}

template <class T>
ndarray<T>::ndarray(
    const std::shared_ptr<std::vector<T>>& data,
    const std::vector<std::size_t>& shape,
    const std::vector<std::size_t>& strides,
    const std::size_t offset)
    : m_ndim(shape.size()), m_shape(shape), m_strides(strides),
      m_size(shape_to_size(shape)), m_offset(offset),
      m_data(
          (data == nullptr) ? std::make_shared<std::vector<T>>(m_size) : data)
{
    if (shape.size() != strides.size()) {
        throw std::invalid_argument(
            "Size of shape does not match with that of strides, "
            + std::to_string(shape.size())
            + " != " + std::to_string(strides.size()));
    }
    if (in(shape, 0UL) || in(strides, 0UL)) {
        throw std::invalid_argument("0 in `shape` or `strides`");
    }
    const auto index = index_of_last_element(shape, strides, offset);
    if (m_data->size() <= index) {
        throw std::invalid_argument("Size of data is too short");
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
