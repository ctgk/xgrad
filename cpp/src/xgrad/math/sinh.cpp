#include <cmath>

#include "xgrad/core/tensor.hpp"
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
tensor<T> sinh(const tensor<T>& x)
{
    const auto& node = internal::get_node(x);
    const std::shared_ptr<internal::operation_node<T>>& op = std::make_shared<
        internal::unary_operation<T, internal::sinh_operation>>(node);
    return internal::create_tensor(op);
}

template tensor<float> sinh<float>(const tensor<float>&);
template tensor<double> sinh<double>(const tensor<double>&);

} // namespace xgrad
