#ifndef XGRAD_MATH_SINH_HPP
#define XGRAD_MATH_SINH_HPP

#include "xgrad/core/ndarray.hpp"

namespace xgrad
{

/**
 * @brief Element-wise sine hypyerbolic function
 *
 * @tparam T
 * Data type, float or double
 * @param x
 * Input array
 * @return ndarray<T>
 * Element-wise sine hyperbolic values.
 */
template <class T>
ndarray<T> sinh(const ndarray<T>& x);

} // namespace xgrad

#endif // XGRAD_MATH_SINH_HPP
