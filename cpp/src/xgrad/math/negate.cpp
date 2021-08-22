#include "xgrad/math/negate.hpp"
#include "xgrad/core/tensor.hpp"

#include "xgrad/core/unary_operation.hpp"

namespace xgrad
{

namespace internal
{

template <class T>
struct negate_operation
{
    struct forward
    {
        T operator()(const T x) const
        {
            return -x;
        }
    };
    struct backward
    {
        T operator()(const T, const T) const
        {
            return static_cast<T>(-1);
        }
    };
};

} // namespace internal

template <class T>
tensor<T> negate(const tensor<T>& x)
{
    const auto& node = internal::get_node(x);
    const std::shared_ptr<internal::operation_node<T>>& op = std::make_shared<
        internal::unary_operation<T, internal::negate_operation>>(node);
    return internal::create_tensor(op);
}

template <class T>
tensor<T> operator-(const tensor<T>& x)
{
    return negate(x);
}

template tensor<float> negate<float>(const tensor<float>&);
template tensor<double> negate<double>(const tensor<double>&);
template tensor<float> operator-<float>(const tensor<float>&);
template tensor<double> operator-<double>(const tensor<double>&);

} // namespace xgrad
