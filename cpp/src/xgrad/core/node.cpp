#include <memory>

#include "xgrad/core/node.hpp"

namespace xgrad
{

void node::add_child(node* const child)
{
    m_children.emplace_back(child);
}

node::node(const std::shared_ptr<node>& parent) : m_parent(parent)
{
    parent->add_child(this);
}

const std::shared_ptr<node>& node::parent() const
{
    return m_parent;
}

} // namespace xgrad
