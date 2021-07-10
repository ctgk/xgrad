#include <cmath>
#include <vector>

#include "xgrad/core.hpp"
#include "xgrad/math.hpp"

#include "test_xgrad/test_backward.hpp"

#include <CppUTest/TestHarness.h>

namespace test_xgrad
{

TEST_GROUP(tanh){};

static const xgrad::ndarray<double> input[] = {
    xgrad::ndarray<double>(xgrad::ndshape({4}), {0, -1, 1, 0.5}),
};

static const xgrad::ndarray<double> expected_forward[] = {
    xgrad::ndarray<double>(
        xgrad::ndshape({4}),
        {0, std::tanh(-1.), std::tanh(1.), std::tanh(0.5)}),
};

static const xgrad::ndarray<double> expected_backward[] = {
    xgrad::ndarray<double>(
        xgrad::ndshape({4}),
        {1 / (std::cosh(0.) * std::cosh(0.)),
         1 / (std::cosh(-1.) * std::cosh(-1.)),
         1 / (std::cosh(1.) * std::cosh(1.)),
         1 / (std::cosh(0.5) * std::cosh(0.5))}),
};

TEST(tanh, forward)
{
    for (auto ii = sizeof(input) / sizeof(xgrad::ndarray<double>); ii--;) {
        const auto actual = xgrad::tanh(input[ii]);
        CHECK_TRUE(xgrad::allclose(actual, expected_forward[ii]));
    }
}

TEST(tanh, backward)
{
    for (auto ii = sizeof(input) / sizeof(xgrad::ndarray<double>); ii--;) {
        test_backward<double>(
            input[ii], xgrad::tanh<double>, expected_backward[ii]);
    }
}

} // namespace test_xgrad
