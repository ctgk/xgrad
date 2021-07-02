#ifndef XGRAD_UTILITY_PRODUCT_HPP
#define XGRAD_UTILITY_PRODUCT_HPP

#include <functional>
#include <numeric>

namespace xgrad::utility
{

template <class T, class InputIterator>
inline T product(InputIterator begin, InputIterator end)
{
    return std::accumulate(
        begin, end, static_cast<T>(1), std::multiplies<T>());
}

} // namespace xgrad::utility

#endif // XGRAD_UTILITY_PRODUCT_HPP
