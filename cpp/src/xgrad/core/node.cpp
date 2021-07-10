#include <initializer_list>
#include <memory>

#include "xgrad/core/node.hpp"
#include "xgrad/utility/shape_to_strides.hpp"

namespace xgrad::internal
{

template <class T>
operation_node<T>::operation_node(
    const std::initializer_list<std::shared_ptr<ndarray_node<T>>>& arguments)
    : m_arguments(arguments)
{
    for (auto&& arg : arguments) {
        arg->increment_num_children();
    }
}

template <class T>
operation_node<T>::~operation_node()
{
    for (auto&& arg : m_arguments) {
        arg->decrement_num_children();
    }
}

template <class T>
const std::vector<std::shared_ptr<ndarray_node<T>>>&
operation_node<T>::arguments() const
{
    return m_arguments;
}

template <class T>
bool operation_node<T>::has_differentiable_arguments() const
{
    for (auto&& arg : m_arguments) {
        if (arg->requires_grad()) {
            return true;
        }
    }
    return false;
}

template <class T>
bool operation_node<T>::differentiable() const
{
    return has_differentiable_arguments();
}

template <class T>
ndshape operation_node<T>::output_shape() const
{
    return ndshape({});
}

template <class T>
std::shared_ptr<std::vector<T>> operation_node<T>::output_data() const
{
    const auto shape = output_shape();
    const auto size = shape.product();
    return std::make_shared<std::vector<T>>(size);
}

template <class T>
std::shared_ptr<std::vector<T>> operation_node<T>::output_grad() const
{
    if (!differentiable()) {
        return nullptr;
    }
    const auto shape = output_shape();
    const auto size = shape.product();
    return std::make_shared<std::vector<T>>(size, static_cast<T>(1));
}

template <class T>
void operation_node<T>::forward(ndarray_node<T>* const out)
{
    for (auto&& arg : m_arguments) {
        arg->reset_num_backward();
    }
    forward_impl(out);
}

template <class T>
void operation_node<T>::forward_impl(ndarray_node<T>* const) const
{
}

template <class T>
void operation_node<T>::backward(const ndarray_node<T>& out)
{
    for (auto&& arg : m_arguments) {
        if (arg->requires_grad() && (arg->num_backward() == 0)
            && !arg->is_view()) {
            std::fill_n(
                arg->grad()->begin(),
                arg->shape().product(),
                static_cast<T>(0));
        }
    }
    backward_impl(out);
    for (auto&& arg : m_arguments) {
        if (arg->requires_grad()) {
            arg->increment_num_backward();
        }
    }
    for (auto&& arg : m_arguments) {
        if (arg->requires_grad()
            && (arg->num_backward() == arg->num_children())) {
            arg->backward();
        }
    }
}

template <class T>
void operation_node<T>::backward_impl(const ndarray_node<T>&) const
{
}

template class operation_node<float>;
template class operation_node<double>;

template <class T>
ndarray_node<T>::ndarray_node() : ndarray_node(ndshape({}), nullptr)
{
}

template <class T>
ndarray_node<T>::ndarray_node(
    const ndshape& shape, const std::shared_ptr<std::vector<T>>& data)
    : computational_graph_node(), m_is_view(false), m_shape(shape),
      m_strides(utility::shape_to_strides(shape)),
      m_data(
          (data == nullptr)
              ? std::make_shared<std::vector<T>>(m_shape.product())
              : data),
      m_grad(nullptr), m_parent(nullptr)
{
}

template <class T>
ndarray_node<T>::ndarray_node(const std::shared_ptr<operation_node<T>>& op)
    : computational_graph_node(), m_is_view(false),
      m_shape(op->output_shape()),
      m_strides(utility::shape_to_strides(m_shape)), m_data(op->output_data()),
      m_grad(op->output_grad()), m_parent(op)
{
    op->increment_num_children();
    op->forward(this);
}

template <class T>
bool ndarray_node<T>::is_view() const
{
    return m_is_view;
}

template <class T>
const ndshape& ndarray_node<T>::shape() const
{
    return m_shape;
}

template <class T>
const std::vector<std::size_t>& ndarray_node<T>::strides() const
{
    return m_strides;
}

template <class T>
const std::shared_ptr<std::vector<T>>& ndarray_node<T>::data() const
{
    return m_data;
}

template <class T>
const std::shared_ptr<std::vector<T>>& ndarray_node<T>::grad() const
{
    return m_grad;
}

template <class T>
bool ndarray_node<T>::requires_grad() const
{
    return m_grad != nullptr;
}

template <class T>
void ndarray_node<T>::requires_grad(const bool flag)
{
    if (flag && m_grad == nullptr) {
        m_grad = std::make_shared<std::vector<T>>(
            m_data->size(), static_cast<T>(1));
    }
    if (!flag) {
        m_grad = nullptr;
        if (m_parent != nullptr) {
            m_parent->decrement_num_children();
            m_parent = nullptr;
        }
    }
}

template <class T>
void ndarray_node<T>::backward()
{
    if (m_parent == nullptr)
        return;
    if (m_parent->num_backward() == 0) {
        m_parent->backward(*this);
    }
}

template class ndarray_node<float>;
template class ndarray_node<double>;

} // namespace xgrad::internal
