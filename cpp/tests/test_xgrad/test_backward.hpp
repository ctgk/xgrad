#ifndef TEST_XGRAD_TEST_BACKWARD_HPP
#define TEST_XGRAD_TEST_BACKWARD_HPP

#include <functional>

#include "xgrad/core.hpp"

namespace test_xgrad
{

template <class T>
bool test_backward(
    const xgrad::ndarray<T>& input,
    const std::function<xgrad::ndarray<T>(const xgrad::ndarray<T>&)>&
        forward_function,
    const xgrad::ndarray<T>& expected)
{
    if (input.is_view()) {
        throw std::runtime_error("");
    }
    auto a = xgrad::ndarray<T>(
        input.shape(),
        std::vector<T>(input.cdata(), input.cdata() + input.size()));
    a.requires_grad(true);
    auto out = forward_function(a);
    out.backward();
    const auto actual = a.grad_to_array();
    return xgrad::allclose(actual, expected);
}

} // namespace test_xgrad

#endif // TEST_XGRAD_TEST_BACKWARD_HPP
