#ifndef XGRAD_MATH_EXP_HPP
#define XGRAD_MATH_EXP_HPP

#include "xgrad/core/ndarray.hpp"

namespace xgrad
{

/**
 * @brief Return element-wise exponentiated array
 *
 * @tparam T
 * Data type, float or double
 * @param x
 * Input array
 * @return ndarray<T>
 * Element-wise exponentiated array
 */
template <class T>
ndarray<T> exp(const ndarray<T>& x);

} // namespace xgrad

#endif // XGRAD_MATH_EXP_HPP
