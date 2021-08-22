#include <cmath>

#include "xgrad/core/tensor.hpp"
#include "xgrad/math/sin.hpp"

#include "xgrad/core/unary_operation.hpp"

namespace xgrad
{

namespace internal
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

} // namespace internal

template <class T>
tensor<T> sin(const tensor<T>& x)
{
    const auto& node = internal::get_node(x);
    const std::shared_ptr<internal::operation_node<T>>& op = std::make_shared<
        internal::unary_operation<T, internal::sin_operation>>(node);
    return internal::create_tensor(op);
}

template tensor<float> sin<float>(const tensor<float>&);
template tensor<double> sin<double>(const tensor<double>&);

} // namespace xgrad
