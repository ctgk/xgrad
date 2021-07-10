#include "xgrad/math/negate.hpp"
#include "xgrad/core/ndarray.hpp"

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
ndarray<T> negate(const ndarray<T>& x)
{
    const auto& node = internal::get_node(x);
    const std::shared_ptr<internal::operation_node<T>>& op = std::make_shared<
        internal::unary_operation<T, internal::negate_operation>>(node);
    return internal::create_ndarray(op);
}

template <class T>
ndarray<T> operator-(const ndarray<T>& x)
{
    return negate(x);
}

template ndarray<float> negate<float>(const ndarray<float>&);
template ndarray<double> negate<double>(const ndarray<double>&);
template ndarray<float> operator-<float>(const ndarray<float>&);
template ndarray<double> operator-<double>(const ndarray<double>&);

} // namespace xgrad
