#include "xgrad/core/ndarray.hpp"
#include "xgrad/math.hpp"

#include "xgrad/core/unary_operation.hpp"
#include "xgrad/math/square.hpp"

namespace xgrad
{

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
