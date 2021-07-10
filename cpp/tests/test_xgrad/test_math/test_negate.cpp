#include <vector>

#include "xgrad/core.hpp"
#include "xgrad/math.hpp"

#include "test_xgrad/test_backward.hpp"

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
        test_backward<float>(
            input[ii], xgrad::negate<float>, expected_backward[ii]);
    }
}

} // namespace test_xgrad
