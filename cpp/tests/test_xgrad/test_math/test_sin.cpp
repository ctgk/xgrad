#include <cmath>
#include <vector>

#include "xgrad/core.hpp"
#include "xgrad/math.hpp"

#include "test_xgrad/test_backward.hpp"

#include <CppUTest/TestHarness.h>

namespace test_xgrad
{

TEST_GROUP(sin){};

static const xgrad::tensor<double> input[] = {
    xgrad::tensor<double>(xgrad::ndshape({4}), {0, M_PI, M_PI_2, -0.5}),
};

static const xgrad::tensor<double> expected_forward[] = {
    xgrad::tensor<double>(xgrad::ndshape({4}), {0, 0, 1, std::sin(-0.5)}),
};

static const xgrad::tensor<double> expected_backward[] = {
    xgrad::tensor<double>(xgrad::ndshape({4}), {1, -1, 0, std::cos(-0.5)}),
};

TEST(sin, forward)
{
    for (auto ii = sizeof(input) / sizeof(xgrad::tensor<double>); ii--;) {
        const auto actual = xgrad::sin(input[ii]);
        CHECK_TRUE(xgrad::allclose(actual, expected_forward[ii]));
    }
}

TEST(sin, backward)
{
    for (auto ii = sizeof(input) / sizeof(xgrad::tensor<double>); ii--;) {
        test_backward<double>(
            input[ii], xgrad::sin<double>, expected_backward[ii]);
    }
}

TEST(sin, numerical_gradient)
{
    for (auto ii = sizeof(input) / sizeof(xgrad::tensor<double>); ii--;) {
        const auto grad = numerical_gradient(xgrad::sin<double>, input[ii]);
        test_backward<double>(input[ii], xgrad::sin<double>, grad);
    }
}

} // namespace test_xgrad
