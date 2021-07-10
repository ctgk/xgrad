#ifndef XGRAD_MATH_NEGATE_HPP
#define XGRAD_MATH_NEGATE_HPP

#include "xgrad/core/ndarray.hpp"

namespace xgrad
{

/**
 * @brief return element-wise negate valued array.
 *
 * @tparam T
 * Data type, float or double
 * @param x
 * Input array
 * @return ndarray<T>
 * Element-wise negate valued array
 */
template <class T>
ndarray<T> negate(const ndarray<T>& x);

/**
 * @brief return element-wise negate valued array.
 *
 * @tparam T
 * Data type, float or double
 * @param x
 * Input array
 * @return ndarray<T>
 * Element-wise negate valued array
 */
template <class T>
ndarray<T> operator-(const ndarray<T>& x);

} // namespace xgrad

#endif // XGRAD_MATH_NEGATE_HPP
