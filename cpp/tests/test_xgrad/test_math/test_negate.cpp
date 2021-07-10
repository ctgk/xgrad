#include <vector>

#include "xgrad/core.hpp"
#include "xgrad/math.hpp"

#include <CppUTest/TestHarness.h>

namespace test_xgrad
{

TEST_GROUP(negate){};

static const xgrad::ndarray<float> input[] = {
    xgrad::ndarray<float>(),
    xgrad::ndarray<float>(xgrad::ndshape({3, 2}), {1, -2, 3, -4, 5, -6}),
};

static const xgrad::ndarray<float> expected_forward[] = {
    xgrad::ndarray<float>(),
    xgrad::ndarray<float>(xgrad::ndshape({3, 2}), {-1, 2, -3, 4, -5, 6}),
};

static const xgrad::ndarray<float> expected_backward[] = {
    xgrad::ndarray<float>(xgrad::ndshape(), {-1}),
    xgrad::ndarray<float>(xgrad::ndshape({3, 2}), {-1, -1, -1, -1, -1, -1}),
};

TEST(negate, forward)
{
    for (auto ii = sizeof(input) / sizeof(xgrad::ndarray<float>); ii--;) {
        const auto actual = xgrad::negate(input[ii]);
        CHECK_TRUE(xgrad::allclose(actual, expected_forward[ii]));
    }
}

TEST(negate, backward)
{
    for (auto ii = sizeof(input) / sizeof(xgrad::ndarray<float>); ii--;) {
        auto a = xgrad::ndarray<float>(
            input[ii].shape(),
            std::vector<float>(
                input[ii].cdata(), input[ii].cdata() + input[ii].size()));
        a.requires_grad(true);
        auto out = xgrad::negate(a);
        out.backward();
        const auto actual = a.grad_to_array();
        CHECK_TRUE(xgrad::allclose(actual, expected_backward[ii]));
    }
}

} // namespace test_xgrad
