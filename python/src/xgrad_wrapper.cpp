#include <fstream>
#include <iostream>
#include <string>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "xgrad/core.hpp"
#include "xgrad/math.hpp"

namespace py = pybind11;
namespace xg = xgrad;

const char* const pointer(const char* const string)
{
    return string;
}

xg::tensor<float> array_to_tensor(
    const py::array_t<float, py::array::c_style | py::array::forcecast>& a)
{
    const auto shape = xg::ndshape(
        std::vector<std::size_t>(a.shape(), a.shape() + a.ndim()));
    const auto data = std::vector<float>(
        reinterpret_cast<float*>(a.request().ptr),
        reinterpret_cast<float*>(a.request().ptr) + a.size());
    return xg::tensor<float>(shape, data);
}

py::array
tensor_to_array(xg::tensor<float>& self, const bool return_data = true)
{
    if (!return_data && !self.requires_grad()) {
        throw std::invalid_argument("Cannot gradient of a constant tensor.");
    }
    if (self.is_view()) {
        throw std::invalid_argument(
            "Cannot get values from a view of a tensor.");
    }
    const std::vector<int> shape(self.shape().cbegin(), self.shape().cend());
    std::vector<int> strides(self.strides().cbegin(), self.strides().cend());
    for (auto&& s : strides) {
        s *= static_cast<int>(sizeof(float));
    }
    auto cap
        = py::capsule(new auto(xg::internal::get_node(self)), [](void* p) {
              delete reinterpret_cast<
                  std::shared_ptr<xg::internal::tensor_node<float>>*>(p);
          });
    return py::array(
        py::buffer_info(
            return_data ? self.data() : self.grad(),
            sizeof(float),
            py::format_descriptor<float>::value,
            self.ndim(),
            shape,
            strides),
        cap);
}

void export_tensor(py::module& m)
{
    py::class_<xg::tensor<float>>(m, "Tensor", (XGRAD_TENSOR_DOCSTRING))
        .def(py::init(&array_to_tensor), (XGRAD_TENSOR_INIT_DOCSTRING))
        .def_property_readonly(
            "is_view",
            &xg::tensor<float>::is_view,
            (XGRAD_TENSOR_IS_VIEW_DOCSTRING))
        .def_property_readonly(
            "ndim", &xg::tensor<float>::ndim, XGRAD_TENSOR_NDIM_DOCSTRING)
        .def_property_readonly(
            "size", &xg::tensor<float>::size, XGRAD_TENSOR_SIZE_DOCSTRING)
        .def_property_readonly(
            "shape",
            [](const xg::tensor<float>& a) {
                return py::cast(std::vector<std::size_t>(
                    a.shape().cbegin(), a.shape().cend()));
            },
            XGRAD_TENSOR_SHAPE_DOCSTRING)
        .def_property_readonly(
            "strides",
            py::overload_cast<>(&xg::tensor<float>::strides, py::const_),
            XGRAD_TENSOR_STRIDES_DOCSTRING)
        .def_property_readonly(
            "data",
            [](xg::tensor<float>& self) { return tensor_to_array(self); },
            XGRAD_TENSOR_DATA_DOCSTRING)
        .def_property_readonly(
            "grad",
            [](xg::tensor<float>& self) {
                return tensor_to_array(self, false);
            },
            XGRAD_TENSOR_GRAD_DOCSTRING)
        .def_property(
            "requires_grad",
            py::overload_cast<>(&xg::tensor<float>::requires_grad, py::const_),
            py::overload_cast<const bool>(&xg::tensor<float>::requires_grad),
            XGRAD_TENSOR_REQUIRES_GRAD_DOCSTRING)
        .def(
            "backward",
            &xg::tensor<float>::backward,
            XGRAD_TENSOR_BACKWARD_DOCSTRING)
        .def(
            "__neg__",
            [](const xg::tensor<float>& self) { return -self; },
            XGRAD_TENSOR_NEG_DOCSTRING);
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
    const char* const docstrings[] = {
        XGRAD_COS_DOCSTRING,
        XGRAD_COSH_DOCSTRING,
        XGRAD_EXP_DOCSTRING,
        XGRAD_LOG_DOCSTRING,
        XGRAD_NEGATE_DOCSTRING,
        XGRAD_SIN_DOCSTRING,
        XGRAD_SINH_DOCSTRING,
        XGRAD_SQUARE_DOCSTRING,
        XGRAD_TAN_DOCSTRING,
        XGRAD_TANH_DOCSTRING,
    };
    for (auto ii = sizeof(unary_operation_str) / sizeof(const char*); ii--;) {
        m.def(unary_operation_str[ii], unary_operation[ii]);
        m.def(
            unary_operation_str[ii],
            [unary_operation,
             ii](const py::array_t<
                 float,
                 py::array::c_style | py::array::forcecast>& a) {
                return unary_operation[ii](array_to_tensor(a));
            },
            docstrings[ii]);
    }
}

PYBIND11_MODULE(_xgrad_cpp, m)
{
    export_tensor(m);
    export_unary_operations(m);
}
