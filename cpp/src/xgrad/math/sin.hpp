#ifndef XGRAD_MATH_SIN_HPP
#define XGRAD_MATH_SIN_HPP

#include <cmath>

namespace xgrad::internal
{

template <class T>
struct sin_operation
{
    struct forward
    {
        T operator()(const T x) const
        {
            return std::sin(x);
        }
    };
    struct backward
    {
        T operator()(const T x, const T) const
        {
            return std::cos(x);
        }
    };
};

} // namespace xgrad::internal

#endif // XGRAD_MATH_SIN_HPP
