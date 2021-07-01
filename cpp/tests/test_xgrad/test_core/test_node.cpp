#include <memory>

#include "xgrad/core.hpp"

#include <CppUTest/TestHarness.h>

namespace test_xgrad
{

TEST_GROUP(node){};

TEST(node, parent)
{
    {
        auto a = xgrad::node();
        CHECK_TRUE(a.parent() == nullptr);
    }
    {
        auto a = std::make_shared<xgrad::node>();
        auto b = xgrad::node(a);
        CHECK_TRUE(b.parent() == a);
    }
}

} // namespace test_xgrad
