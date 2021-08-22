#include <vector>

#include "xgrad/core.hpp"
#include "xgrad/math.hpp"

#include "test_xgrad/test_backward.hpp"

#include <CppUTest/TestHarness.h>

namespace test_xgrad
{

TEST_GROUP(square){};

static const xgrad::tensor<double> input[] = {
    xgrad::tensor<double>(
        xgrad::ndshape({2, 4}), {1, -2, 3, -4, 5, -6, 7, -8}),
};

static const xgrad::tensor<double> expected_forward[] = {
    xgrad::tensor<double>(
        xgrad::ndshape({2, 4}), {1, 4, 9, 16, 25, 36, 49, 64}),
};

static const xgrad::tensor<double> expected_backward[] = {
    xgrad::tensor<double>(
        xgrad::ndshape({2, 4}), {2, -4, 6, -8, 10, -12, 14, -16}),
};

TEST(square, forward)
{
    for (auto ii = sizeof(input) / sizeof(xgrad::tensor<double>); ii--;) {
        const auto actual = xgrad::square(input[ii]);
        CHECK_TRUE(xgrad::allclose(actual, expected_forward[ii]));
    }
}

TEST(square, backward)
{
    for (auto ii = sizeof(input) / sizeof(xgrad::tensor<double>); ii--;) {
        test_backward<double>(
            input[ii], xgrad::square<double>, expected_backward[ii]);
    }
}

TEST(square, numerical_gradient)
{
    for (auto ii = sizeof(input) / sizeof(xgrad::tensor<double>); ii--;) {
        const auto grad = numerical_gradient(xgrad::square<double>, input[ii]);
        test_backward<double>(input[ii], xgrad::square<double>, grad);
    }
}

} // namespace test_xgrad
