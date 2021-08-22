#ifndef XGRAD_MATH_EXP_HPP
#define XGRAD_MATH_EXP_HPP

#include "xgrad/core/tensor.hpp"

namespace xgrad
{

/**
 * @brief Return element-wise exponentiated array
 *
 * @tparam T
 * Data type, float or double
 * @param x
 * Input array
 * @return tensor<T>
 * Element-wise exponentiated array
 */
template <class T>
tensor<T> exp(const tensor<T>& x);

} // namespace xgrad

#endif // XGRAD_MATH_EXP_HPP
