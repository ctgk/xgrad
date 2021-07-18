#include <initializer_list>
#include <memory>

#include "xgrad/core/tensor.hpp"

#include <CppUTest/TestHarness.h>

namespace test_xgrad
{

static xgrad::tensor<float> array[] = {
    xgrad::tensor<float>(),
    xgrad::tensor<float>(xgrad::ndshape({2})),
    xgrad::tensor<float>(xgrad::ndshape({3, 4})),
};

static const std::size_t expected_ndim[] = {
    0UL,
    1UL,
    2UL,
};

static const std::vector<std::size_t> expected_shape[] = {
    std::vector<std::size_t>(),
    std::vector<std::size_t>({2}),
    std::vector<std::size_t>({3, 4}),
};

static const std::vector<std::size_t> expected_strides[] = {
    std::vector<std::size_t>(),
    std::vector<std::size_t>({1}),
    std::vector<std::size_t>({4, 1}),
};

static std::size_t axis[] = {0UL, 0UL, 1UL};
static const long int expected_length[] = {-1, 2, 4};
static const long int expected_stride[] = {-1, 1, 1};

TEST_GROUP(tensor){};

TEST(tensor, init)
{
    xgrad::tensor<float>(
        xgrad::ndshape({2UL, 3UL, 2UL}),
        std::make_shared<std::vector<float>>(12UL, 1.f));
    CHECK_THROWS(
        std::invalid_argument,
        xgrad::tensor<float>(
            xgrad::ndshape({2UL, 3UL, 2UL}),
            std::make_shared<std::vector<float>>(13UL, 0.f)));
}

TEST(tensor, ndim)
{
    for (auto i = sizeof(array) / sizeof(xgrad::tensor<float>); i--;) {
        CHECK_EQUAL(expected_ndim[i], array[i].ndim());
    }
}

TEST(tensor, shape)
{
    for (auto i = sizeof(array) / sizeof(xgrad::tensor<float>); i--;) {
        CHECK_TRUE(array[i].shape() == expected_shape[i]);
    }
}

TEST(tensor, shape_axis)
{
    for (auto i = sizeof(array) / sizeof(xgrad::tensor<float>); i--;) {
        if (expected_length[i] < 0) {
            CHECK_THROWS(std::out_of_range, array[i].shape(axis[i]));
        } else {
            CHECK_EQUAL(
                static_cast<std::size_t>(expected_length[i]),
                array[i].shape(axis[i]));
        }
    }
}

TEST(tensor, strides)
{
    for (auto i = sizeof(array) / sizeof(xgrad::tensor<float>); i--;) {
        CHECK_TRUE(array[i].strides() == expected_strides[i]);
    }
}

TEST(tensor, strides_axis)
{
    for (auto i = sizeof(array) / sizeof(xgrad::tensor<float>); i--;) {
        if (expected_stride[i] < 0) {
            CHECK_THROWS(std::out_of_range, array[i].strides(axis[i]));
        } else {
            CHECK_EQUAL(
                static_cast<std::size_t>(expected_stride[i]),
                array[i].strides(axis[i]));
        }
    }
}

} // namespace test_xgrad
