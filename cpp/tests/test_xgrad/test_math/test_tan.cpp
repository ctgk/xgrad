#include <cmath>
#include <vector>

#include "xgrad/core.hpp"
#include "xgrad/math.hpp"

#include "test_xgrad/test_backward.hpp"

#include <CppUTest/TestHarness.h>

namespace test_xgrad
{

TEST_GROUP(tan){};

static const xgrad::ndarray<double> input[] = {
    xgrad::ndarray<double>(xgrad::ndshape({4}), {0, M_PI_4, -0.5, -M_PI / 6}),
};

static const xgrad::ndarray<double> expected_forward[] = {
    xgrad::ndarray<double>(
        xgrad::ndshape({4}), {0, 1, std::tan(-0.5), std::tan(-M_PI / 6)}),
};

static const xgrad::ndarray<double> expected_backward[] = {
    xgrad::ndarray<double>(
        xgrad::ndshape({4}),
        {0, 0.5, 1 / (std::cos(-0.5) * std::cos(-0.5)), 0.25}),
};

TEST(tan, forward)
{
    for (auto ii = sizeof(input) / sizeof(xgrad::ndarray<double>); ii--;) {
        const auto actual = xgrad::tan(input[ii]);
        CHECK_TRUE(xgrad::allclose(actual, expected_forward[ii]));
    }
}

TEST(tan, backward)
{
    for (auto ii = sizeof(input) / sizeof(xgrad::ndarray<double>); ii--;) {
        test_backward<double>(
            input[ii], xgrad::tan<double>, expected_backward[ii]);
    }
}

TEST(tan, numerical_gradient)
{
    for (auto ii = sizeof(input) / sizeof(xgrad::ndarray<double>); ii--;) {
        const auto grad = numerical_gradient(xgrad::tan<double>, input[ii]);
        test_backward<double>(input[ii], xgrad::tan<double>, grad);
    }
}

} // namespace test_xgrad
