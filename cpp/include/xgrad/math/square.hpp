#ifndef XGRAD_MATH_SQUARE_HPP
#define XGRAD_MATH_SQUARE_HPP

#include "xgrad/core/ndarray.hpp"

namespace xgrad
{

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

#endif // XGRAD_MATH_SQUARE_HPP
