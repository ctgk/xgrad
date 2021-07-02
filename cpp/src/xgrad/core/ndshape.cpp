#include <functional>
#include <numeric>

#include "xgrad/core/ndshape.hpp"

#include "xgrad/utility/product.hpp"

namespace xgrad
{

template <class InputIterator>
static inline std::array<std::size_t, k_max_ndim>
to_array(InputIterator begin, InputIterator end, const std::size_t offset)
{
    auto a = std::array<std::size_t, k_max_ndim>();
    std::copy(begin, end, a.data() + offset);
    return a;
}

ndshape::ndshape(const std::size_t ndim, const std::size_t* const begin)
    : m_ndim(ndim), m_offset(k_max_ndim - ndim),
      m_product(utility::product<std::size_t>(begin, begin + ndim)),
      m_data(to_array(begin, begin + ndim, m_offset))
{
    if (ndim > k_max_ndim) {
        throw std::invalid_argument(
            std::to_string(ndim) + " exceeds the maximum dimensionality "
            + std::to_string(k_max_ndim));
    }
}

ndshape::ndshape() : ndshape({})
{
}

ndshape::ndshape(const std::initializer_list<std::size_t>& shape)
    : ndshape(shape.size(), shape.begin())
{
}

ndshape::ndshape(const std::vector<std::size_t>& shape)
    : ndshape(shape.size(), shape.data())
{
}

std::size_t ndshape::ndim() const
{
    return m_ndim;
}

std::size_t ndshape::product() const
{
    return m_product;
}

std::size_t ndshape::operator[](const std::size_t axis) const
{
    if (axis >= m_ndim) {
        throw std::out_of_range(
            "Axis " + std::to_string(axis) + " exceeds the dimesionality "
            + std::to_string(m_ndim));
    }
    return m_data[m_offset + axis];
}

std::size_t ndshape::operator[](const int axis) const
{
    return ndshape::operator[](
        (axis < 0) ? static_cast<std::size_t>(axis + static_cast<int>(m_ndim))
                   : static_cast<std::size_t>(axis));
}

bool ndshape::operator!=(const ndshape& other) const
{
    if (m_ndim != other.m_ndim) {
        return true;
    }
    for (auto it1 = crbegin(), it2 = other.crbegin(); it1 != crend();
         ++it1, ++it2) {
        if (*it1 != *it2) {
            return true;
        }
    }
    return false;
}

bool ndshape::operator==(const ndshape& other) const
{
    return !(*this != other);
}

std::array<std::size_t, k_max_ndim>::const_iterator ndshape::cbegin() const
{
    return m_data.cbegin() + m_offset;
}

std::array<std::size_t, k_max_ndim>::const_iterator ndshape::cend() const
{
    return m_data.cend();
}

std::array<std::size_t, k_max_ndim>::const_reverse_iterator
ndshape::crbegin() const
{
    return m_data.crbegin();
}

std::array<std::size_t, k_max_ndim>::const_reverse_iterator
ndshape::crend() const
{
    return m_data.crbegin()
           + static_cast<
               std::reverse_iterator<const std::size_t*>::difference_type>(
               m_ndim);
}

} // namespace xgrad
