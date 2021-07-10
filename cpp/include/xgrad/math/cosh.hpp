#ifndef XGRAD_MATH_COSH_HPP
#define XGRAD_MATH_COSH_HPP

#include "xgrad/core/ndarray.hpp"

namespace xgrad
{

/**
 * @brief Element-wise cosine hypyerbolic function
 *
 * @tparam T
 * Data type, float or double
 * @param x
 * Input array
 * @return ndarray<T>
 * Element-wise cosine hyperbolic values.
 */
template <class T>
ndarray<T> cosh(const ndarray<T>& x);

} // namespace xgrad

#endif // XGRAD_MATH_COSH_HPP
