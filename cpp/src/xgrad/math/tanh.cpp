#include <cmath>

#include "xgrad/core/ndarray.hpp"
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
ndarray<T> tanh(const ndarray<T>& x)
{
    const auto& node = internal::get_node(x);
    const std::shared_ptr<internal::operation_node<T>>& op = std::make_shared<
        internal::unary_operation<T, internal::tanh_operation>>(node);
    return internal::create_ndarray(op);
}

template ndarray<float> tanh<float>(const ndarray<float>&);
template ndarray<double> tanh<double>(const ndarray<double>&);

} // namespace xgrad
