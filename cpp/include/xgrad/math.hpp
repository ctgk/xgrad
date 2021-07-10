#ifndef XGRAD_MATH_HPP
#define XGRAD_MATH_HPP

#include "xgrad/core/ndarray.hpp"

namespace xgrad
{

template <class T>
ndarray<T> negate(const ndarray<T>& x);

template <class T>
ndarray<T> operator-(const ndarray<T>& x);

} // namespace xgrad

#endif // XGRAD_MATH_HPP
