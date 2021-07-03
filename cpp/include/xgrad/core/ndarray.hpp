#ifndef XGRAD_CORE_NDARRAY_HPP
#define XGRAD_CORE_NDARRAY_HPP

#include <memory>
#include <type_traits>

#include "xgrad/core/ndshape.hpp"

namespace xgrad
{

namespace internal
{

template <class T>
class ndarray_node;

} // namespace internal

template <class T>
class ndarray
{
    static_assert(
        std::is_same<T, float>::value || std::is_same<T, double>::value,
        "Only xgrad::ndarray<float> or xgrad::ndarray<double> is allowed.");

private:
    const std::shared_ptr<internal::ndarray_node<T>> m_node;

public:
    ndarray();
    ndarray(
        const ndshape& shape,
        const std::shared_ptr<std::vector<T>>& data = nullptr);

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
     * @return const ndshape&
     * Shape fo this array.
     */
    const ndshape& shape() const;

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

    /**
     * @brief Return pointer to data.
     *
     * @return T*
     * Pointer to data.
     */
    T* data();

    /**
     * @brief Return pointer to data
     *
     * @return const T*
     * Pointer to data
     */
    const T* cdata() const;

    /**
     * @brief Return pointer to grad.
     *
     * @return T*
     * Pointer to grad.
     */
    T* grad();

    /**
     * @brief Return pointer to grad.
     *
     * @return const T*
     * Pointer to grad.
     */
    const T* cgrad() const;

    /**
     * @brief Return a flag whether the array is variable.
     *
     * @return true The array is variable.
     * @return false The array is not variable.
     */
    bool requires_grad() const;

    /**
     * @brief Make the array variable or constant.
     *
     * @param flag
     * The array becomes variable if true, otherwise becomes constant.
     */
    void requires_grad(const bool flag);

    /**
     * @brief Backpropagate gradient through computational graph.
     *
     */
    void backward();
};

} // namespace xgrad

#endif // XGRAD_CORE_NDARRAY_HPP
