#ifndef XGRAD_MATH_SIN_HPP
#define XGRAD_MATH_SIN_HPP

#include "xgrad/core/tensor.hpp"

namespace xgrad
{

/**
 * @brief Element-wise sinusoidal function
 *
 * @tparam T
 * Data type, float or double
 * @param x
 * Input array
 * @return tensor<T>
 * Element-wise sin values.
 */
template <class T>
tensor<T> sin(const tensor<T>& x);

} // namespace xgrad

#endif // XGRAD_MATH_SIN_HPP
