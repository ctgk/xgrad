#ifndef XGRAD_MATH_NEGATE_HPP
#define XGRAD_MATH_NEGATE_HPP

namespace xgrad::internal
{

template <class T>
struct negate_operation
{
    struct forward
    {
        T operator()(const T x) const
        {
            return -x;
        }
    };
    struct backward
    {
        T operator()(const T dy, const T, const T) const
        {
            return -dy;
        }
    };
};

} // namespace xgrad::internal

#endif // XGRAD_MATH_NEGATE_HPP
