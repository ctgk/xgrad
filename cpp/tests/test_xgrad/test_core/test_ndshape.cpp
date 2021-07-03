#include "xgrad/core.hpp"

#include <CppUTest/TestHarness.h>

namespace test_xgrad
{

TEST_GROUP(ndshape){};

TEST(ndshape, init)
{
    CHECK_THROWS(
        std::invalid_argument, xgrad::ndshape({1, 2, 3, 4, 5, 6, 7, 8, 9}));
}

TEST(ndshape, ndim)
{
    CHECK_EQUAL(0UL, xgrad::ndshape().ndim());
    CHECK_EQUAL(1UL, xgrad::ndshape({1}).ndim());
    CHECK_EQUAL(2UL, xgrad::ndshape({4, 3}).ndim());
    CHECK_EQUAL(3UL, xgrad::ndshape({2, 5, 3}).ndim());
    CHECK_EQUAL(4UL, xgrad::ndshape({2, 5, 1, 8}).ndim());
}

TEST(ndshape, product)
{
    CHECK_EQUAL(1UL, xgrad::ndshape().product());
    CHECK_EQUAL(1UL, xgrad::ndshape({1}).product());
    CHECK_EQUAL(12UL, xgrad::ndshape({4, 3}).product());
    CHECK_EQUAL(30UL, xgrad::ndshape({2, 5, 3}).product());
    CHECK_EQUAL(80UL, xgrad::ndshape({2, 5, 1, 8}).product());
}

TEST(ndshape, getitem)
{
    CHECK_THROWS(std::out_of_range, xgrad::ndshape()[0]);
    CHECK_THROWS(std::out_of_range, xgrad::ndshape({1})[1]);
    CHECK_THROWS(std::out_of_range, xgrad::ndshape({4, 3})[-3]);
    CHECK_THROWS(std::out_of_range, xgrad::ndshape({2, 5, 3})[3]);
    CHECK_THROWS(std::out_of_range, xgrad::ndshape({2, 5, 1, 8})[9]);

    CHECK_EQUAL(1, xgrad::ndshape({1})[0]);
    CHECK_EQUAL(3, xgrad::ndshape({4, 3})[-1]);
    CHECK_EQUAL(3, xgrad::ndshape({2, 5, 3})[2]);
    CHECK_EQUAL(1, xgrad::ndshape({2, 5, 1, 8})[-2]);
}

TEST(ndshape, not_equal)
{
    CHECK_TRUE(xgrad::ndshape({2, 3}) != xgrad::ndshape({2}));
    CHECK_TRUE(xgrad::ndshape({}) != xgrad::ndshape({2}));
}

TEST(ndshape, equal)
{
    CHECK_TRUE(xgrad::ndshape({2, 3}) == xgrad::ndshape({2, 3}));
    CHECK_TRUE(xgrad::ndshape({}) == xgrad::ndshape({}));
}

TEST(ndshape, const_iterator)
{
    {
        const auto s = xgrad::ndshape();
        CHECK_TRUE(s.cbegin() == s.cend());
    }
    {
        const auto s = xgrad::ndshape({4, 7, 2});
        auto it = s.cbegin();
        CHECK_EQUAL(4, *it);
        ++it;
        CHECK_EQUAL(7, *it);
        ++it;
        CHECK_EQUAL(2, *it);
        ++it;
        CHECK_TRUE(it == s.cend());
    }
}

TEST(ndshape, const_reverse_iterator)
{
    {
        const auto s = xgrad::ndshape();
        CHECK_TRUE(s.crbegin() == s.crend());
    }
    {
        const auto s = xgrad::ndshape({4, 7, 2});
        auto it = s.crbegin();
        CHECK_EQUAL(2, *it);
        ++it;
        CHECK_EQUAL(7, *it);
        ++it;
        CHECK_EQUAL(4, *it);
        ++it;
        CHECK_TRUE(it == s.crend());
    }
}

} // namespace test_xgrad
