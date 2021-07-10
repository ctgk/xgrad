#include "xgrad/core/ndarray.hpp"
#include "xgrad/math.hpp"

#include "xgrad/core/unary_operation.hpp"
#include "xgrad/math/negate.hpp"

namespace xgrad
{

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
