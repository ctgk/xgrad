#include <cmath>

#include "xgrad/core/ndarray.hpp"
#include "xgrad/math/sinh.hpp"

#include "xgrad/core/unary_operation.hpp"

namespace xgrad
{

namespace internal
{

template <class T>
struct sinh_operation
{
    struct forward
    {
        T operator()(const T x) const
        {
            return std::sinh(x);
        }
    };
    struct backward
    {
        T operator()(const T x, const T) const
        {
            return std::cosh(x);
        }
    };
};

} // namespace internal

template <class T>
ndarray<T> sinh(const ndarray<T>& x)
{
    const auto& node = internal::get_node(x);
    const std::shared_ptr<internal::operation_node<T>>& op = std::make_shared<
        internal::unary_operation<T, internal::sinh_operation>>(node);
    return internal::create_ndarray(op);
}

template ndarray<float> sinh<float>(const ndarray<float>&);
template ndarray<double> sinh<double>(const ndarray<double>&);

} // namespace xgrad
