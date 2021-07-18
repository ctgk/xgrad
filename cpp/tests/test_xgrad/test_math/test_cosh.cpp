#include <cmath>
#include <vector>

#include "xgrad/core.hpp"
#include "xgrad/math.hpp"

#include "test_xgrad/test_backward.hpp"

#include <CppUTest/TestHarness.h>

namespace test_xgrad
{

TEST_GROUP(cosh){};

static const xgrad::tensor<double> input[] = {
    xgrad::tensor<double>(xgrad::ndshape({4}), {0, -1, 1, 0.5}),
};

static const xgrad::tensor<double> expected_forward[] = {
    xgrad::tensor<double>(
        xgrad::ndshape({4}),
        {1, std::cosh(-1.), std::cosh(1.), std::cosh(0.5)}),
};

static const xgrad::tensor<double> expected_backward[] = {
    xgrad::tensor<double>(
        xgrad::ndshape({4}),
        {0, std::sinh(-1.), std::sinh(1.), std::sinh(0.5)}),
};

TEST(cosh, forward)
{
    for (auto ii = sizeof(input) / sizeof(xgrad::tensor<double>); ii--;) {
        const auto actual = xgrad::cosh(input[ii]);
        CHECK_TRUE(xgrad::allclose(actual, expected_forward[ii]));
    }
}

TEST(cosh, backward)
{
    for (auto ii = sizeof(input) / sizeof(xgrad::tensor<double>); ii--;) {
        test_backward<double>(
            input[ii], xgrad::cosh<double>, expected_backward[ii]);
    }
}

TEST(cosh, numerical_gradient)
{
    for (auto ii = sizeof(input) / sizeof(xgrad::tensor<double>); ii--;) {
        const auto grad = numerical_gradient(xgrad::cosh<double>, input[ii]);
        test_backward<double>(input[ii], xgrad::cosh<double>, grad);
    }
}

} // namespace test_xgrad
