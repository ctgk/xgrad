#include <cmath>

#include "xgrad/core/tensor.hpp"
#include "xgrad/math/tan.hpp"

#include "xgrad/core/unary_operation.hpp"

namespace xgrad
{

namespace internal
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

} // namespace internal

template <class T>
tensor<T> tan(const tensor<T>& x)
{
    const auto& node = internal::get_node(x);
    const std::shared_ptr<internal::operation_node<T>>& op = std::make_shared<
        internal::unary_operation<T, internal::tan_operation>>(node);
    return internal::create_tensor(op);
}

template tensor<float> tan<float>(const tensor<float>&);
template tensor<double> tan<double>(const tensor<double>&);

} // namespace xgrad
