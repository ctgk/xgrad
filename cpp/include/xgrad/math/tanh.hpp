#ifndef XGRAD_MATH_TANH_HPP
#define XGRAD_MATH_TANH_HPP

#include "xgrad/core/ndarray.hpp"

namespace xgrad
{

/**
 * @brief Element-wise tangent hypyerbolic function
 *
 * @tparam T
 * Data type, float or double
 * @param x
 * Input array
 * @return ndarray<T>
 * Element-wise tangent hyperbolic values.
 */
template <class T>
ndarray<T> tanh(const ndarray<T>& x);

} // namespace xgrad

#endif // XGRAD_MATH_TANH_HPP
