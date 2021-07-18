#ifndef XGRAD_MATH_SINH_HPP
#define XGRAD_MATH_SINH_HPP

#include "xgrad/core/tensor.hpp"

namespace xgrad
{

/**
 * @brief Element-wise sine hypyerbolic function
 *
 * @tparam T
 * Data type, float or double
 * @param x
 * Input array
 * @return tensor<T>
 * Element-wise sine hyperbolic values.
 */
template <class T>
tensor<T> sinh(const tensor<T>& x);

} // namespace xgrad

#endif // XGRAD_MATH_SINH_HPP
