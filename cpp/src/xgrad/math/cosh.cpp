#include <cmath>

#include "xgrad/core/tensor.hpp"
#include "xgrad/math/cosh.hpp"

#include "xgrad/core/unary_operation.hpp"

namespace xgrad
{

namespace internal
{

template <class T>
struct cosh_operation
{
    struct forward
    {
        T operator()(const T x) const
        {
            return std::cosh(x);
        }
    };
    struct backward
    {
        T operator()(const T x, const T) const
        {
            return std::sinh(x);
        }
    };
};

} // namespace internal

template <class T>
tensor<T> cosh(const tensor<T>& x)
{
    const auto& node = internal::get_node(x);
    const std::shared_ptr<internal::operation_node<T>>& op = std::make_shared<
        internal::unary_operation<T, internal::cosh_operation>>(node);
    return internal::create_tensor(op);
}

template tensor<float> cosh<float>(const tensor<float>&);
template tensor<double> cosh<double>(const tensor<double>&);

} // namespace xgrad
