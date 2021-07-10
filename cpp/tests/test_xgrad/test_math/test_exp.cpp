#include <cmath>
#include <vector>

#include "xgrad/core.hpp"
#include "xgrad/math.hpp"

#include "test_xgrad/test_backward.hpp"

#include <CppUTest/TestHarness.h>

namespace test_xgrad
{

TEST_GROUP(exp){};

static const xgrad::ndarray<double> input[] = {
    xgrad::ndarray<double>(xgrad::ndshape({4}), {0, -1, 2, 0.5}),
};

static const xgrad::ndarray<double> expected_forward[] = {
    xgrad::ndarray<double>(
        xgrad::ndshape({4}), {1, std::exp(-1.), std::exp(2.), std::exp(0.5)}),
};

static const xgrad::ndarray<double> expected_backward[] = {
    xgrad::ndarray<double>(
        xgrad::ndshape({4}), {1, std::exp(-1.), std::exp(2.), std::exp(0.5)}),
};

TEST(exp, forward)
{
    for (auto ii = sizeof(input) / sizeof(xgrad::ndarray<double>); ii--;) {
        const auto actual = xgrad::exp(input[ii]);
        CHECK_TRUE(xgrad::allclose(actual, expected_forward[ii]));
    }
}

TEST(exp, backward)
{
    for (auto ii = sizeof(input) / sizeof(xgrad::ndarray<double>); ii--;) {
        test_backward<double>(
            input[ii], xgrad::exp<double>, expected_backward[ii]);
    }
}

} // namespace test_xgrad
