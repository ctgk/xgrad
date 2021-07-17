#ifndef XGRAD_CORE_TENSOR_HPP
#define XGRAD_CORE_TENSOR_HPP

#include <memory>
#include <type_traits>

#include "xgrad/core/ndshape.hpp"

namespace xgrad
{

template <class T>
class tensor;

namespace internal
{

template <class T>
class tensor_node;

template <class T>
class operation_node;

template <class T>
const std::shared_ptr<tensor_node<T>>& get_node(const tensor<T>& a);

template <class T>
tensor<T> create_tensor(const std::shared_ptr<operation_node<T>>& op);

} // namespace internal

template <class T>
class tensor
{
    static_assert(
        std::is_same<T, float>::value || std::is_same<T, double>::value,
        "Only xgrad::tensor<float> or xgrad::tensor<double> is allowed.");

private:
    const std::shared_ptr<internal::tensor_node<T>> m_node;

    tensor(const std::shared_ptr<internal::operation_node<T>>& op);

    friend const std::shared_ptr<internal::tensor_node<T>>&
    internal::get_node<T>(const tensor<T>&);
    friend tensor<T> internal::create_tensor<T>(
        const std::shared_ptr<internal::operation_node<T>>& op);

public:
    tensor();
    tensor(
        const ndshape& shape,
        const std::shared_ptr<std::vector<T>>& data = nullptr);
    tensor(const ndshape& shape, const std::vector<T>& data);

    /**
     * @brief Return whether if the array is a view of another.
     *
     * @return true The array is a view of another.
     * @return false The array is not a view of another.
     */
    bool is_view() const;

    /**
     * @brief Return dimensionality of this array.
     *
     * @return std::size_t
     * Dimensionality of this array.
     */
    std::size_t ndim() const;

    /**
     * @brief Return number of element contained in this array.
     *
     * @return std::size_t
     * Number of element contained in this array.
     */
    std::size_t size() const;

    /**
     * @brief Return shape of this array.
     *
     * @return const ndshape&
     * Shape fo this array.
     */
    const ndshape& shape() const;

    /**
     * @brief Return length of the array along the axis.
     *
     * @param axis
     * Axis to count its length along.
     * @return std::size_t
     * Length of the array along the axis.
     */
    std::size_t shape(const std::size_t axis) const;

    /**
     * @brief Return strides of the array
     *
     * @return const std::vector<std::size_t>
     * Strides of the array.
     */
    const std::vector<std::size_t>& strides() const;

    /**
     * @brief Return strides along the axis.
     *
     * @param axis
     * Axis of strides.
     * @return std::size_t
     * Strides along the axis.
     */
    std::size_t strides(const std::size_t axis) const;

    /**
     * @brief Return pointer to data.
     *
     * @return T*
     * Pointer to data.
     */
    T* data();

    /**
     * @brief Return pointer to data
     *
     * @return const T*
     * Pointer to data
     */
    const T* cdata() const;

    /**
     * @brief Return pointer to grad.
     *
     * @return T*
     * Pointer to grad.
     */
    T* grad();

    /**
     * @brief Return pointer to grad.
     *
     * @return const T*
     * Pointer to grad.
     */
    const T* cgrad() const;

    /**
     * @brief Return tensor of grad values.
     *
     * @return tensor<T>
     * tensor of grad values.
     */
    tensor<T> grad_to_array() const;

    /**
     * @brief Return a flag whether the array is variable.
     *
     * @return true The array is variable.
     * @return false The array is not variable.
     */
    bool requires_grad() const;

    /**
     * @brief Make the array variable or constant.
     *
     * @param flag
     * The array becomes variable if true, otherwise becomes constant.
     */
    void requires_grad(const bool flag);

    /**
     * @brief Backpropagate gradient through computational graph.
     *
     */
    void backward();
};

} // namespace xgrad

#endif // XGRAD_CORE_TENSOR_HPP
