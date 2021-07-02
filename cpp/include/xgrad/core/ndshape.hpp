#ifndef XGRAD_CORE_NDSHAPE_HPP
#define XGRAD_CORE_NDSHAPE_HPP

#include <array>
#include <vector>

namespace xgrad
{

constexpr std::size_t k_max_ndim = 8;

class ndshape
{
private:
    using base_type = std::array<std::size_t, k_max_ndim>;

    const std::size_t m_ndim;
    const std::size_t m_offset;
    const std::size_t m_product;
    const std::array<std::size_t, k_max_ndim> m_data;

    ndshape(const std::size_t ndim, const std::size_t* const begin);

public:
    ndshape();
    ndshape(const std::initializer_list<std::size_t>& shape);
    ndshape(const std::vector<std::size_t>& shape);
    std::size_t ndim() const;
    std::size_t product() const;
    std::size_t operator[](const std::size_t axis) const;
    std::size_t operator[](const int axis) const;
    bool operator!=(const ndshape& other) const;
    bool operator==(const ndshape& other) const;
    std::array<std::size_t, k_max_ndim>::const_iterator cbegin() const;
    std::array<std::size_t, k_max_ndim>::const_iterator cend() const;
    std::array<std::size_t, k_max_ndim>::const_reverse_iterator
    crbegin() const;
    std::array<std::size_t, k_max_ndim>::const_reverse_iterator crend() const;
};

} // namespace xgrad

#endif // XGRAD_CORE_NDSHAPE_HPP
