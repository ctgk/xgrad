#ifndef XGRAD_UTILITY_SHAPE_TO_STRIDES_HPP
#define XGRAD_UTILITY_SHAPE_TO_STRIDES_HPP

#include <vector>

#include "xgrad/core/ndshape.hpp"

namespace xgrad
{

namespace utility
{

inline std::vector<std::size_t> shape_to_strides(const ndshape& shape)
{
    if (shape.ndim() == 0UL) {
        return std::vector<std::size_t>();
    }
    auto strides = std::vector<std::size_t>(shape.ndim(), 1UL);
    for (auto i = shape.ndim() - 1UL; i--;) {
        strides[i] = strides[i + 1UL] * shape[i + 1UL];
    }
    return strides;
}

} // namespace utility

} // namespace xgrad

#endif // XGRAD_UTILITY_SHAPE_TO_STRIDES_HPP
