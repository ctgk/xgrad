#ifndef XGRAD_MATH_COS_HPP
#define XGRAD_MATH_COS_HPP

#include "xgrad/core/ndarray.hpp"

namespace xgrad
{

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

} // namespace xgrad

#endif // XGRAD_MATH_COS_HPP
