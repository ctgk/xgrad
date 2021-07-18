#include <vector>

#include "xgrad/core.hpp"
#include "xgrad/math.hpp"

#include "test_xgrad/test_backward.hpp"

#include <CppUTest/TestHarness.h>

namespace test_xgrad
{

TEST_GROUP(negate){};

static const xgrad::tensor<float> input[] = {
    xgrad::tensor<float>(),
    xgrad::tensor<float>(xgrad::ndshape({3, 2}), {1, -2, 3, -4, 5, -6}),
};

static const xgrad::tensor<float> expected_forward[] = {
    xgrad::tensor<float>(),
    xgrad::tensor<float>(xgrad::ndshape({3, 2}), {-1, 2, -3, 4, -5, 6}),
};

static const xgrad::tensor<float> expected_backward[] = {
    xgrad::tensor<float>(xgrad::ndshape(), {-1}),
    xgrad::tensor<float>(xgrad::ndshape({3, 2}), {-1, -1, -1, -1, -1, -1}),
};

TEST(negate, forward)
{
    for (auto ii = sizeof(input) / sizeof(xgrad::tensor<float>); ii--;) {
        const auto actual = xgrad::negate(input[ii]);
        CHECK_TRUE(xgrad::allclose(actual, expected_forward[ii]));
    }
}

TEST(negate, backward)
{
    for (auto ii = sizeof(input) / sizeof(xgrad::tensor<float>); ii--;) {
        test_backward<float>(
            input[ii], xgrad::negate<float>, expected_backward[ii]);
    }
}

TEST(negate, numerical_gradient)
{
    for (auto ii = sizeof(input) / sizeof(xgrad::tensor<float>); ii--;) {
        const auto a = xgrad::tensor<double>(
            input[ii].shape(),
            std::vector<double>(
                input[ii].cdata(), input[ii].cdata() + input[ii].size()));
        const auto grad = numerical_gradient(xgrad::negate<double>, a);
        test_backward<double>(a, xgrad::negate<double>, grad);
    }
}

} // namespace test_xgrad
