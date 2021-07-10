#ifndef XGRAD_MATH_SIN_HPP
#define XGRAD_MATH_SIN_HPP

#include "xgrad/core/ndarray.hpp"

namespace xgrad
{

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

} // namespace xgrad

#endif // XGRAD_MATH_SIN_HPP
