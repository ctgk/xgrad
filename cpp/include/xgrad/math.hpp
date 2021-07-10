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

} // namespace xgrad

#endif // XGRAD_MATH_HPP
