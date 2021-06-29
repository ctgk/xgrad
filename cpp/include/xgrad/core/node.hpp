#ifndef XGRAD_CORE_NODE_HPP
#define XGRAD_CORE_NODE_HPP

#include <memory>
#include <vector>

namespace xgrad
{

class node
{
private:
    const std::shared_ptr<node> m_parent;
    std::vector<node*> m_children;
    void add_child(node* const child);

public:
    node() = default;
    node(const std::shared_ptr<node>& parent);
    const std::shared_ptr<node>& parent() const;
};

} // namespace xgrad

#endif // XGRAD_CORE_NODE_HPP
