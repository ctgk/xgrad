#include "xgrad/math/square.hpp"
#include "xgrad/core/tensor.hpp"

#include "xgrad/core/unary_operation.hpp"

namespace xgrad
{

namespace internal
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

} // namespace internal

template <class T>
tensor<T> square(const tensor<T>& x)
{
    const auto& node = internal::get_node(x);
    const std::shared_ptr<internal::operation_node<T>>& op = std::make_shared<
        internal::unary_operation<T, internal::square_operation>>(node);
    return internal::create_tensor(op);
}

template tensor<float> square<float>(const tensor<float>&);
template tensor<double> square<double>(const tensor<double>&);

} // namespace xgrad
