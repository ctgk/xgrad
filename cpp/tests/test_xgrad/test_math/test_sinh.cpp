#include <cmath>
#include <vector>

#include "xgrad/core.hpp"
#include "xgrad/math.hpp"

#include "test_xgrad/test_backward.hpp"

#include <CppUTest/TestHarness.h>

namespace test_xgrad
{

TEST_GROUP(sinh){};

static const xgrad::ndarray<double> input[] = {
    xgrad::ndarray<double>(xgrad::ndshape({4}), {0, -1, 1, 0.5}),
};

static const xgrad::ndarray<double> expected_forward[] = {
    xgrad::ndarray<double>(
        xgrad::ndshape({4}),
        {0, std::sinh(-1.), std::sinh(1.), std::sinh(0.5)}),
};

static const xgrad::ndarray<double> expected_backward[] = {
    xgrad::ndarray<double>(
        xgrad::ndshape({4}),
        {1, std::cosh(-1.), std::cosh(1.), std::cosh(0.5)}),
};

TEST(sinh, forward)
{
    for (auto ii = sizeof(input) / sizeof(xgrad::ndarray<double>); ii--;) {
        const auto actual = xgrad::sinh(input[ii]);
        CHECK_TRUE(xgrad::allclose(actual, expected_forward[ii]));
    }
}

TEST(sinh, backward)
{
    for (auto ii = sizeof(input) / sizeof(xgrad::ndarray<double>); ii--;) {
        test_backward<double>(
            input[ii], xgrad::sinh<double>, expected_backward[ii]);
    }
}

TEST(sinh, numerical_gradient)
{
    for (auto ii = sizeof(input) / sizeof(xgrad::ndarray<double>); ii--;) {
        const auto grad = numerical_gradient(xgrad::sinh<double>, input[ii]);
        test_backward<double>(input[ii], xgrad::sinh<double>, grad);
    }
}

} // namespace test_xgrad
