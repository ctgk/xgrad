#include <cmath>
#include <vector>

#include "xgrad/core.hpp"
#include "xgrad/math.hpp"

#include "test_xgrad/test_backward.hpp"

#include <CppUTest/TestHarness.h>

namespace test_xgrad
{

TEST_GROUP(log){};

static const xgrad::tensor<double> input[] = {
    xgrad::tensor<double>(xgrad::ndshape({4}), {0.1, 1, 2, 0.5}),
};

static const xgrad::tensor<double> expected_forward[] = {
    xgrad::tensor<double>(
        xgrad::ndshape({4}),
        {std::log(0.1), std::log(1.), std::log(2.), std::log(0.5)}),
};

static const xgrad::tensor<double> expected_backward[] = {
    xgrad::tensor<double>(xgrad::ndshape({4}), {10, 1, 0.5, 2}),
};

TEST(log, forward)
{
    for (auto ii = sizeof(input) / sizeof(xgrad::tensor<double>); ii--;) {
        const auto actual = xgrad::log(input[ii]);
        CHECK_TRUE(xgrad::allclose(actual, expected_forward[ii]));
    }
}

TEST(log, backward)
{
    for (auto ii = sizeof(input) / sizeof(xgrad::tensor<double>); ii--;) {
        test_backward<double>(
            input[ii], xgrad::log<double>, expected_backward[ii]);
    }
}

TEST(log, numerical_gradient)
{
    for (auto ii = sizeof(input) / sizeof(xgrad::tensor<double>); ii--;) {
        const auto grad = numerical_gradient(xgrad::log<double>, input[ii]);
        test_backward<double>(input[ii], xgrad::log<double>, grad);
    }
}

} // namespace test_xgrad
