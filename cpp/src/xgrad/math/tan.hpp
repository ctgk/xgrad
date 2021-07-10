#ifndef XGRAD_MATH_TAN_HPP
#define XGRAD_MATH_TAN_HPP

#include <cmath>

namespace xgrad::internal
{

template <class T>
struct tan_operation
{
    struct forward
    {
        T operator()(const T x) const
        {
            return std::tan(x);
        }
    };
    struct backward
    {
        T operator()(const T x, const T) const
        {
            const auto c = std::cos(x);
            return 1 / (c * c);
        }
    };
};

} // namespace xgrad::internal

#endif // XGRAD_MATH_TAN_HPP
