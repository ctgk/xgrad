#include <cmath>

#include "xgrad/core/tensor.hpp"
#include "xgrad/math/log.hpp"

#include "xgrad/core/unary_operation.hpp"

namespace xgrad
{

namespace internal
{

template <class T>
struct log_operation
{
    struct forward
    {
        T operator()(const T x) const
        {
            return std::log(x);
        }
    };
    struct backward
    {
        T operator()(const T x, const T) const
        {
            return 1 / x;
        }
    };
};

} // namespace internal

template <class T>
tensor<T> log(const tensor<T>& x)
{
    const auto& node = internal::get_node(x);
    const std::shared_ptr<internal::operation_node<T>>& op = std::make_shared<
        internal::unary_operation<T, internal::log_operation>>(node);
    return internal::create_tensor(op);
}

template tensor<float> log<float>(const tensor<float>&);
template tensor<double> log<double>(const tensor<double>&);

} // namespace xgrad
