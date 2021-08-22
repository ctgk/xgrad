#ifndef XGRAD_CORE_UNARY_OPERATION_HPP
#define XGRAD_CORE_UNARY_OPERATION_HPP

#include <algorithm>

#include "xgrad/core/node.hpp"

namespace xgrad::internal
{

template <class T, template <class U> class Operation>
class unary_operation final : public operation_node<T>
{
private:
    typename Operation<T>::forward m_forward;
    typename Operation<T>::backward m_backward;

public:
    unary_operation(const std::shared_ptr<tensor_node<T>>& x)
        : operation_node<T>({x}), m_forward(), m_backward()
    {
    }
    unary_operation(
        const std::shared_ptr<tensor_node<T>>& x,
        typename Operation<T>::forward f,
        typename Operation<T>::backward df)
        : operation_node<T>({x}, m_forward(f), m_backward(df))
    {
    }
    ndshape output_shape() const final
    {
        return this->m_arguments[0]->shape();
    }
    void forward_impl(tensor_node<T>* const out) const final
    {
        const tensor_node<T>& a = *this->m_arguments[0];
        if (!a.is_view()) {
            std::transform(
                a.data()->cbegin(),
                a.data()->cend(),
                out->data()->begin(),
                m_forward);
        } else {
            throw std::runtime_error("Argument of unary_operation must not be "
                                     "a view of another array.");
        }
    }
    void backward_impl(const tensor_node<T>& out) const final
    {
        const tensor_node<T>& a = *this->m_arguments[0];
        if (!a.is_view()) {
            const auto size = a.shape().product();
            const T* x = a.data()->data();
            T* dx = a.grad()->data();
            const T* y = out.data()->data();
            const T* dy = out.grad()->data();
            for (auto ii = size; ii--;) {
                dx[ii] += dy[ii] * m_backward(x[ii], y[ii]);
            }
        } else {
            throw std::runtime_error("Argument of unary_operation must not be "
                                     "a view of another array.");
        }
    }
};

} // namespace xgrad::internal

#endif // XGRAD_CORE_UNARY_OPERATION_HPP
