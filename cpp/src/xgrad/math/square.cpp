#include "xgrad/math/square.hpp"
#include "xgrad/core/ndarray.hpp"

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
ndarray<T> square(const ndarray<T>& x)
{
    const auto& node = internal::get_node(x);
    const std::shared_ptr<internal::operation_node<T>>& op = std::make_shared<
        internal::unary_operation<T, internal::square_operation>>(node);
    return internal::create_ndarray(op);
}

template ndarray<float> square<float>(const ndarray<float>&);
template ndarray<double> square<double>(const ndarray<double>&);

} // namespace xgrad
