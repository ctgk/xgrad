#ifndef XGRAD_MATH_TAN_HPP
#define XGRAD_MATH_TAN_HPP

#include "xgrad/core/ndarray.hpp"

namespace xgrad
{

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

#endif // XGRAD_MATH_TAN_HPP
