#ifndef XGRAD_MATH_SQUARE_HPP
#define XGRAD_MATH_SQUARE_HPP

namespace xgrad::internal
{

template <class T>
struct square_operation
{
    struct forward
    {
        T operator()(const T x) const
        {
            return x * x;
        }
    };
    struct backward
    {
        T operator()(const T x, const T) const
        {
            return x + x;
        }
    };
};

} // namespace xgrad::internal

#endif // XGRAD_MATH_SQUARE_HPP
