#ifndef XGRAD_MATH_NEGATE_HPP
#define XGRAD_MATH_NEGATE_HPP

#include "xgrad/core/tensor.hpp"

namespace xgrad
{

/**
 * @brief return element-wise negate valued array.
 *
 * @tparam T
 * Data type, float or double
 * @param x
 * Input array
 * @return tensor<T>
 * Element-wise negate valued array
 */
template <class T>
tensor<T> negate(const tensor<T>& x);

/**
 * @brief return element-wise negate valued array.
 *
 * @tparam T
 * Data type, float or double
 * @param x
 * Input array
 * @return tensor<T>
 * Element-wise negate valued array
 */
template <class T>
tensor<T> operator-(const tensor<T>& x);

} // namespace xgrad

#endif // XGRAD_MATH_NEGATE_HPP
