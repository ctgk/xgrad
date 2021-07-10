#ifndef TEST_XGRAD_TEST_BACKWARD_HPP
#define TEST_XGRAD_TEST_BACKWARD_HPP

#include <functional>
#include <numeric>

#include "xgrad/core.hpp"

namespace test_xgrad
{

template <class Func>
inline xgrad::ndarray<double> numerical_gradient(
    Func function, const xgrad::ndarray<double>& arg, const double eps = 1e-5)
{
    if (arg.is_view()) {
        throw std::runtime_error("");
    }
    auto grad = xgrad::ndarray<double>(arg.shape());
    for (auto ii = arg.size(); ii--;) {
        auto p = xgrad::ndarray<double>(
            arg.shape(),
            std::vector<double>(arg.cdata(), arg.cdata() + arg.size()));
        p.data()[ii] += eps;
        auto m = xgrad::ndarray<double>(
            arg.shape(),
            std::vector<double>(arg.cdata(), arg.cdata() + arg.size()));
        m.data()[ii] -= eps;
        const auto out_p = function(p);
        const auto out_m = function(m);
        const auto sum_p
            = std::accumulate(out_p.cdata(), out_p.cdata() + out_p.size(), 0.);
        const auto sum_m
            = std::accumulate(out_m.cdata(), out_m.cdata() + out_m.size(), 0.);
        grad.data()[ii] = (sum_p - sum_m) / (2. * eps);
    }
    return grad;
}

template <class T>
inline bool test_backward(
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
