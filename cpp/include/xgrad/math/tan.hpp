#ifndef XGRAD_MATH_TAN_HPP
#define XGRAD_MATH_TAN_HPP

#include "xgrad/core/tensor.hpp"

namespace xgrad
{

/**
 * @brief Element-wise tangent function
 *
 * @tparam T
 * Data type, float or double
 * @param x
 * Input array
 * @return tensor<T>
 * Element-wise tan values
 */
template <class T>
tensor<T> tan(const tensor<T>& x);

} // namespace xgrad

#endif // XGRAD_MATH_TAN_HPP
