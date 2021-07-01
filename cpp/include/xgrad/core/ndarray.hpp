#ifndef XGRAD_CORE_NDARRAY_HPP
#define XGRAD_CORE_NDARRAY_HPP

#include <memory>
#include <type_traits>

#include "xgrad/core/node.hpp"

namespace xgrad
{

template <class T>
class ndarray
{
    static_assert(
        std::is_same<T, float>::value || std::is_same<T, double>::value,
        "Only xgrad::ndarray<float> or xgrad::ndarray<double> is allowed.");

private:
    const std::size_t m_ndim; //!< Dimensionality of the array.
    const std::vector<std::size_t> m_shape;
    const std::vector<std::size_t> m_strides;
    const std::size_t m_size;
    const std::size_t m_offset;
    const std::shared_ptr<std::vector<T>> m_data;

public:
    ndarray();
    ndarray(
        const std::shared_ptr<std::vector<T>>& data,
        const std::vector<std::size_t>& shape);
    ndarray(
        const std::shared_ptr<std::vector<T>>& data,
        const std::vector<std::size_t>& shape,
        const std::vector<std::size_t>& strides,
        const std::size_t offset = 0UL);

    /**
     * @brief Return dimensionality of this array.
     *
     * @return std::size_t
     * Dimensionality of this array.
     */
    std::size_t ndim() const;

    /**
     * @brief Return shape of this array.
     *
     * @return const std::vector<std::size_t>
     * Shape fo this array.
     */
    const std::vector<std::size_t>& shape() const;

    /**
     * @brief Return length of the array along the axis.
     *
     * @param axis
     * Axis to count its length along.
     * @return std::size_t
     * Length of the array along the axis.
     */
    std::size_t shape(const std::size_t axis) const;

    /**
     * @brief Return strides of the array
     *
     * @return const std::vector<std::size_t>
     * Strides of the array.
     */
    const std::vector<std::size_t>& strides() const;

    /**
     * @brief Return strides along the axis.
     *
     * @param axis
     * Axis of strides.
     * @return std::size_t
     * Strides along the axis.
     */
    std::size_t strides(const std::size_t axis) const;
};

} // namespace xgrad

#endif // XGRAD_CORE_NDARRAY_HPP
