#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "xgrad/core.hpp"
#include "xgrad/math.hpp"

namespace py = pybind11;
namespace xg = xgrad;

xg::tensor<float> ndarray_to_tensor(
    const py::array_t<float, py::array::c_style | py::array::forcecast>& a)
{
    const auto shape = xg::ndshape(
        std::vector<std::size_t>(a.shape(), a.shape() + a.ndim()));
    const auto data = std::vector<float>(
        reinterpret_cast<float*>(a.request().ptr),
        reinterpret_cast<float*>(a.request().ptr) + a.size());
    return xg::tensor<float>(shape, data);
}

void export_tensor(py::module& m)
{
    py::class_<xg::tensor<float>>(
        m, "Tensor", "Class for N-dimensional tensor")
        .def(py::init(&ndarray_to_tensor))
        .def_property_readonly("is_view", &xg::tensor<float>::is_view)
        .def_property_readonly(
            "ndim", &xg::tensor<float>::ndim, "Dimesionality of the tensor.")
        .def_property_readonly(
            "size",
            &xg::tensor<float>::size,
            "Number of elements in the tensor.")
        .def_property_readonly(
            "shape",
            [](const xg::tensor<float>& a) {
                return py::cast(std::vector<std::size_t>(
                    a.shape().cbegin(), a.shape().cend()));
            })
        .def_property_readonly(
            "strides",
            py::overload_cast<>(&xg::tensor<float>::strides, py::const_))
        .def_property_readonly(
            "data",
            [](xg::tensor<float>& self) {
                if (self.is_view()) {
                    throw std::invalid_argument(
                        "Converting a view of a tensor to numpy array is not "
                        "supported.");
                }
                const std::vector<int> shape(
                    self.shape().cbegin(), self.shape().cend());
                std::vector<int> strides(
                    self.strides().cbegin(), self.strides().cend());
                for (auto&& s : strides) {
                    s *= static_cast<int>(sizeof(float));
                }
                auto cap = py::capsule(
                    new auto(xg::internal::get_node(self)), [](void* p) {
                        delete reinterpret_cast<std::shared_ptr<
                            xg::internal::tensor_node<float>>*>(p);
                    });
                return py::array(
                    py::buffer_info(
                        self.data(),
                        sizeof(float),
                        py::format_descriptor<float>::value,
                        self.ndim(),
                        shape,
                        strides),
                    cap);
            })
        .def_property_readonly(
            "grad",
            [](xg::tensor<float>& self) {
                if (!self.requires_grad()) {
                    throw std::invalid_argument(
                        "The tensor is not a variable.");
                }
                if (self.is_view()) {
                    throw std::invalid_argument(
                        "Cannot get grad of a view of a tensor.");
                }
                const std::vector<int> shape(
                    self.shape().cbegin(), self.shape().cend());
                std::vector<int> strides(
                    self.strides().cbegin(), self.strides().cend());
                for (auto&& s : strides) {
                    s *= static_cast<int>(sizeof(float));
                }
                auto cap = py::capsule(
                    new auto(xg::internal::get_node(self)), [](void* p) {
                        delete reinterpret_cast<std::shared_ptr<
                            xg::internal::tensor_node<float>>*>(p);
                    });
                return py::array(
                    py::buffer_info(
                        self.grad(),
                        sizeof(float),
                        py::format_descriptor<float>::value,
                        self.ndim(),
                        shape,
                        strides),
                    cap);
            })
        .def_property(
            "requires_grad",
            py::overload_cast<>(&xg::tensor<float>::requires_grad, py::const_),
            py::overload_cast<const bool>(&xg::tensor<float>::requires_grad))
        .def("backward", &xg::tensor<float>::backward)
        .def("__neg__", [](const xg::tensor<float>& self) { return -self; });
}

void export_unary_operations(py::module& m)
{
    const char* const unary_operation_str[]
        = {"cos",
           "cosh",
           "exp",
           "log",
           "negate",
           "sin",
           "sinh",
           "square",
           "tan",
           "tanh"};
    xg::tensor<float> (*unary_operation[])(const xg::tensor<float>&)
        = {xg::cos<float>,
           xg::cosh<float>,
           xg::exp<float>,
           xg::log<float>,
           xg::negate<float>,
           xg::sin<float>,
           xg::sinh<float>,
           xg::square<float>,
           xg::tan<float>,
           xg::tanh<float>};
    for (auto ii = sizeof(unary_operation_str) / sizeof(const char*); ii--;) {
        m.def(unary_operation_str[ii], unary_operation[ii]);
        m.def(
            unary_operation_str[ii],
            [unary_operation,
             ii](const py::array_t<
                 float,
                 py::array::c_style | py::array::forcecast>& a) {
                return unary_operation[ii](ndarray_to_tensor(a));
            });
    }
}

PYBIND11_MODULE(_xgrad_cpp, m)
{
    export_tensor(m);
    export_unary_operations(m);
}
