#ifndef XGRAD_MATH_SQUARE_HPP
#define XGRAD_MATH_SQUARE_HPP

#include "xgrad/core/tensor.hpp"

namespace xgrad
{

/**
 * @brief return element-wise squared values.
 *
 * @tparam T
 * Desired data type, float or double.
 * @param x
 * Input array.
 * @return tensor<T>
 * Array with element-wise squared values.
 */
template <class T>
tensor<T> square(const tensor<T>& x);

} // namespace xgrad

#endif // XGRAD_MATH_SQUARE_HPP
