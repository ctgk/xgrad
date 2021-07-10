#ifndef XGRAD_MATH_HPP
#define XGRAD_MATH_HPP

#include "xgrad/core/ndarray.hpp"

namespace xgrad
{

template <class T>
ndarray<T> negate(const ndarray<T>& x);

template <class T>
ndarray<T> operator-(const ndarray<T>& x);

/**
 * @brief return element-wise squared values.
 *
 * @tparam T
 * Desired data type, float or double.
 * @param x
 * Input array.
 * @return ndarray<T>
 * Array with element-wise squared values.
 */
template <class T>
ndarray<T> square(const ndarray<T>& x);

// trigonometric functions

/**
 * @brief Element-wise sinusoidal function
 *
 * @tparam T
 * Data type, float or double
 * @param x
 * Input array
 * @return ndarray<T>
 * Element-wise sin values.
 */
template <class T>
ndarray<T> sin(const ndarray<T>& x);

/**
 * @brief Element-wise cosine function
 *
 * @tparam T
 * Data type, float or double
 * @param x
 * Input array
 * @return ndarray<T>
 * Element-wise cos values
 */
template <class T>
ndarray<T> cos(const ndarray<T>& x);

/**
 * @brief Element-wise tangent function
 *
 * @tparam T
 * Data type, float or double
 * @param x
 * Input array
 * @return ndarray<T>
 * Element-wise tan values
 */
template <class T>
ndarray<T> tan(const ndarray<T>& x);

} // namespace xgrad

#endif // XGRAD_MATH_HPP
