#include <cmath>

#include "xgrad/core/ndarray.hpp"
#include "xgrad/math/cos.hpp"

#include "xgrad/core/unary_operation.hpp"

namespace xgrad
{

namespace internal
{

template <class T>
struct cos_operation
{
    struct forward
    {
        T operator()(const T x) const
        {
            return std::cos(x);
        }
    };
    struct backward
    {
        T operator()(const T x, const T) const
        {
            return -std::sin(x);
        }
    };
};

} // namespace internal

template <class T>
ndarray<T> cos(const ndarray<T>& x)
{
    const auto& node = internal::get_node(x);
    const std::shared_ptr<internal::operation_node<T>>& op = std::make_shared<
        internal::unary_operation<T, internal::cos_operation>>(node);
    return internal::create_ndarray(op);
}

template ndarray<float> cos<float>(const ndarray<float>&);
template ndarray<double> cos<double>(const ndarray<double>&);

} // namespace xgrad
