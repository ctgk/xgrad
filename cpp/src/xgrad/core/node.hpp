#ifndef XGRAD_CORE_NOE_IMPL_HPP
#define XGRAD_CORE_NOE_IMPL_HPP

#include <initializer_list>
#include <memory>

#include "xgrad/core/ndarray.hpp"
#include "xgrad/core/ndshape.hpp"

namespace xgrad::internal
{

class computational_graph_node
{
private:
    std::size_t m_num_children;
    std::size_t m_num_backward;

public:
    computational_graph_node() = default;

    /**
     * @brief Return number of nodes under this node.
     *
     * @return std::size_t
     * Number of nodes under this node.
     */
    std::size_t num_children() const
    {
        return m_num_children;
    }
    void increment_num_children()
    {
        ++m_num_children;
    }
    void decrement_num_children()
    {
        --m_num_children;
    }
    /**
     * @brief Return number of backprops from child nodes to this node so far.
     *
     * @return std::size_t
     * Number of backprops from child nodes to this node so far.
     */
    std::size_t num_backward() const
    {
        return m_num_backward;
    }
    void increment_num_backward()
    {
        ++m_num_backward;
    }
    void reset_num_backward()
    {
        m_num_backward = 0UL;
    }
};

template <class T>
class operation_node : public computational_graph_node
{
protected:
    const std::vector<std::shared_ptr<ndarray_node<T>>> m_arguments;

public:
    operation_node() = default;
    operation_node(
        const std::initializer_list<std::shared_ptr<ndarray_node<T>>>&
            arguments);
    virtual ~operation_node();
    const std::vector<std::shared_ptr<ndarray_node<T>>>& arguments() const;
    bool has_differentiable_arguments() const;
    virtual bool differentiable() const;
    virtual ndshape output_shape() const;
    virtual std::shared_ptr<std::vector<T>> output_data() const;
    virtual std::shared_ptr<std::vector<T>> output_grad() const;
    void forward(ndarray_node<T>* const out);
    virtual void forward_impl(ndarray_node<T>* const) const;
    void backward(const ndarray_node<T>& out);
    virtual void backward_impl(const ndarray_node<T>& out) const;
};

template <class T>
class ndarray_node : public computational_graph_node
{
private:
    const bool m_is_view; //!< true if an array is a view of another array.
    const ndshape m_shape;
    const std::vector<std::size_t> m_strides;
    const std::shared_ptr<std::vector<T>> m_data;
    std::shared_ptr<std::vector<T>> m_grad;
    std::shared_ptr<operation_node<T>> m_parent;

public:
    ndarray_node();
    ndarray_node(
        const ndshape& shape,
        const std::shared_ptr<std::vector<T>>& data = nullptr);
    ndarray_node(const std::shared_ptr<operation_node<T>>& op);

    bool is_view() const;
    const ndshape& shape() const;
    const std::vector<std::size_t>& strides() const;
    const std::shared_ptr<std::vector<T>>& data() const;
    const std::shared_ptr<std::vector<T>>& grad() const;

    bool requires_grad() const;
    void requires_grad(const bool flag);
    void backward();
};

} // namespace xgrad::internal

#endif // XGRAD_CORE_NOE_IMPL_HPP
