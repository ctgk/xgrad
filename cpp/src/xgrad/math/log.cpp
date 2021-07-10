#include <cmath>

#include "xgrad/core/ndarray.hpp"
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
ndarray<T> log(const ndarray<T>& x)
{
    const auto& node = internal::get_node(x);
    const std::shared_ptr<internal::operation_node<T>>& op = std::make_shared<
        internal::unary_operation<T, internal::log_operation>>(node);
    return internal::create_ndarray(op);
}

template ndarray<float> log<float>(const ndarray<float>&);
template ndarray<double> log<double>(const ndarray<double>&);

} // namespace xgrad
