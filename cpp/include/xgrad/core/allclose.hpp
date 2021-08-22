#ifndef XGRAD_CORE_ALLCLOSE_HPP
#define XGRAD_CORE_ALLCLOSE_HPP

#include "xgrad/core/tensor.hpp"

namespace xgrad
{

template <class T>
bool allclose(
    const tensor<T>& a,
    const tensor<T>& b,
    const T rtol = static_cast<T>(1e-5),
    const T atol = static_cast<T>(1e-8));

} // namespace xgrad

#endif // XGRAD_CORE_ALLCLOSE_HPP
