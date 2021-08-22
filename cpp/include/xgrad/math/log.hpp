#ifndef XGRAD_MATH_LOG_HPP
#define XGRAD_MATH_LOG_HPP

#include "xgrad/core/tensor.hpp"

namespace xgrad
{

/**
 * @brief Return element-wise natural logarithm of the input array
 *
 * @tparam T
 * Data type, float or double
 * @param x
 * Input array
 * @return tensor<T>
 * Element-wise natural logarithm of the input array
 */
template <class T>
tensor<T> log(const tensor<T>& x);

} // namespace xgrad

#endif // XGRAD_MATH_LOG_HPP
