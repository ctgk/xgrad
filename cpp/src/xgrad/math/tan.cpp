#include "xgrad/core/ndarray.hpp"
#include "xgrad/math.hpp"

#include "xgrad/core/unary_operation.hpp"
#include "xgrad/math/tan.hpp"

namespace xgrad
{

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
