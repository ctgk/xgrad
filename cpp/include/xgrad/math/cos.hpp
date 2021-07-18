#ifndef XGRAD_MATH_COS_HPP
#define XGRAD_MATH_COS_HPP

#include "xgrad/core/tensor.hpp"

namespace xgrad
{

/**
 * @brief Element-wise cosine function
 *
 * @tparam T
 * Data type, float or double
 * @param x
 * Input array
 * @return tensor<T>
 * Element-wise cos values
 */
template <class T>
tensor<T> cos(const tensor<T>& x);

} // namespace xgrad

#endif // XGRAD_MATH_COS_HPP
