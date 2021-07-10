#include <cmath>

#include "xgrad/core/ndarray.hpp"
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
ndarray<T> tan(const ndarray<T>& x)
{
    const auto& node = internal::get_node(x);
    const std::shared_ptr<internal::operation_node<T>>& op = std::make_shared<
        internal::unary_operation<T, internal::tan_operation>>(node);
    return internal::create_ndarray(op);
}

template ndarray<float> tan<float>(const ndarray<float>&);
template ndarray<double> tan<double>(const ndarray<double>&);

} // namespace xgrad
