#ifndef XGRAD_MATH_TANH_HPP
#define XGRAD_MATH_TANH_HPP

#include "xgrad/core/tensor.hpp"

namespace xgrad
{

/**
 * @brief Element-wise tangent hypyerbolic function
 *
 * @tparam T
 * Data type, float or double
 * @param x
 * Input array
 * @return tensor<T>
 * Element-wise tangent hyperbolic values.
 */
template <class T>
tensor<T> tanh(const tensor<T>& x);

} // namespace xgrad

#endif // XGRAD_MATH_TANH_HPP
