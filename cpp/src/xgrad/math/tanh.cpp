#include <cmath>

#include "xgrad/core/tensor.hpp"
#include "xgrad/math/tanh.hpp"

#include "xgrad/core/unary_operation.hpp"

namespace xgrad
{

namespace internal
{

template <class T>
struct tanh_operation
{
    struct forward
    {
        T operator()(const T x) const
        {
            return std::tanh(x);
        }
    };
    struct backward
    {
        T operator()(const T x, const T) const
        {
            const auto c = std::cosh(x);
            return 1 / (c * c);
        }
    };
};

} // namespace internal

template <class T>
tensor<T> tanh(const tensor<T>& x)
{
    const auto& node = internal::get_node(x);
    const std::shared_ptr<internal::operation_node<T>>& op = std::make_shared<
        internal::unary_operation<T, internal::tanh_operation>>(node);
    return internal::create_tensor(op);
}

template tensor<float> tanh<float>(const tensor<float>&);
template tensor<double> tanh<double>(const tensor<double>&);

} // namespace xgrad
