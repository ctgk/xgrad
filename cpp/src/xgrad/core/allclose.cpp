#include <cmath>

#include "xgrad/core/allclose.hpp"

namespace xgrad
{

template <class T>
bool allclose(
    const tensor<T>& a, const tensor<T>& b, const T rtol, const T atol)
{
    if (a.shape() != b.shape()) {
        return false;
    }
    if (a.is_view() || b.is_view()) {
        throw std::runtime_error(
            "allclose is not supported for views of arrays");
    }
    const auto a_ptr = a.cdata();
    const auto b_ptr = b.cdata();
    for (auto ii = a.size(); ii--;) {
        if (std::abs(a_ptr[ii] - b_ptr[ii])
            > std::abs(b_ptr[ii]) * rtol + atol) {
            return false;
        }
    }
    return true;
}

template bool
allclose(const tensor<float>&, const tensor<float>&, const float, const float);
template bool allclose(
    const tensor<double>&, const tensor<double>&, const double, const double);

} // namespace xgrad
