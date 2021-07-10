#ifndef XGRAD_MATH_COS_HPP
#define XGRAD_MATH_COS_HPP

#include <cmath>

namespace xgrad::internal
{

template <class T>
struct cos_operation
{
    struct forward
    {
        T operator()(const T x) const
        {
            return std::cos(x);
        }
    };
    struct backward
    {
        T operator()(const T x, const T) const
        {
            return -std::sin(x);
        }
    };
};

} // namespace xgrad::internal

#endif // XGRAD_MATH_COS_HPP
