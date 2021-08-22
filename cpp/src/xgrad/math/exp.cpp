#include <cmath>

#include "xgrad/core/tensor.hpp"
#include "xgrad/math/exp.hpp"

#include "xgrad/core/unary_operation.hpp"

namespace xgrad
{

namespace internal
{

template <class T>
struct exp_operation
{
    struct forward
    {
        T operator()(const T x) const
        {
            return std::exp(x);
        }
    };
    struct backward
    {
        T operator()(const T, const T y) const
        {
            return y;
        }
    };
};

} // namespace internal

template <class T>
tensor<T> exp(const tensor<T>& x)
{
    const auto& node = internal::get_node(x);
    const std::shared_ptr<internal::operation_node<T>>& op = std::make_shared<
        internal::unary_operation<T, internal::exp_operation>>(node);
    return internal::create_tensor(op);
}

template tensor<float> exp<float>(const tensor<float>&);
template tensor<double> exp<double>(const tensor<double>&);

} // namespace xgrad
