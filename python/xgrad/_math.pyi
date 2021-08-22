import typing as tp

import numpy as np

from xgrad._tensor import Tensor


TensorLike = tp.Union[np.ndarray, Tensor]


def cos(arg0: TensorLike) -> Tensor:
    """Return element-wise cosine value of the tensor object.

    Parameters
    ----------
    arg0 : TensorLike
        Input tensor

    Returns
    -------
    Tensor
        Element-wise cosine value of the tensor.

    Examples
    --------
    >>> xg.cos([0, 7 * np.pi / 3])
    array([0. , 0.5], dtype=float32)

    """


def cosh(arg0: TensorLike) -> Tensor:
    """Return element-wise hyperbolic cosine value of the tensor object.

    Parameters
    ----------
    arg0 : TensorLike
        Input tensor

    Returns
    -------
    Tensor
        Element-wise hyperbolic cosine value of the tensor.

    Examples
    --------
    >>> xg.cosh([-100, 0, 100])
    array([inf,  0., inf], dtype=float32)

    """


def exp(arg0: TensorLike) -> Tensor:
    """Return element-wise exponential of the input tensor.

    Parameters
    ----------
    arg0 : TensorLike
        Input tensor object.

    Returns
    -------
    Tensor
        Element-wise exponential of the input tensor.

    Examples
    --------
    >>> xg.exp([0, np.log(2)])
    array([1., 2.], dtype=float32)

    """


def log(arg0: TensorLike) -> Tensor:
    """Return element-wise natural logarithm of the input tensor.

    Parameters
    ----------
    arg0 : TensorLike
        Input tensor object.

    Returns
    -------
    Tensor
        Element-wise natural logarithm of the input tensor.

    Examples
    --------
    >>> xg.log([np.e ** 3, np.e ** 2])
    array([3., 2.], dtype=float32)

    """


def negate(arg0: TensorLike) -> Tensor:
    """Return element-wise negation of the input tensor.

    Parameters
    ----------
    arg0 : TensorLike
        Input tensor object.

    Returns
    -------
    Tensor
        Element-wise negation of the input tensor.

    Examples
    --------
    >>> xg.negate([1, -2, 3])
    array([-1.,  2., -3.], dtype=float32)

    """


def sin(arg0: TensorLike) -> Tensor:
    """Return element-wise sine value of the tensor object.

    Parameters
    ----------
    arg0 : TensorLike
        Input tensor

    Returns
    -------
    Tensor
        Element-wise sine value of the tensor.

    Examples
    --------
    >>> xg.sin([0, -np.pi / 6])
    array([ 0. , -0.5], dtype=float32)

    """


def sinh(arg0: TensorLike) -> Tensor:
    """Return element-wise hyperbolic sine value of the tensor object.

    Parameters
    ----------
    arg0 : TensorLike
        Input tensor

    Returns
    -------
    Tensor
        Element-wise hyperbolic sine value of the tensor.

    Examples
    --------
    >>> xg.sinh([-100, 0, 100])
    array([-inf,   0.,  inf], dtype=float32)

    """


def square(arg0: TensorLike) -> Tensor:
    """Return element-wise square of the input tensor.

    Parameters
    ----------
    arg0 : TensorLike
        Input tensor object.

    Returns
    -------
    Tensor
        Element-wise square of the input tensor.

    Examples
    --------
    >>> xg.square([1, -2, 3])
    array([1., 4., 9.], dtype=float32)

    """


def tan(arg0: TensorLike) -> Tensor:
    """Return element-wise tangent value of the tensor object.

    Parameters
    ----------
    arg0 : TensorLike
        Input tensor

    Returns
    -------
    Tensor
        Element-wise tangent value of the tensor.

    Examples
    --------
    >>> xg.tan([0, np.pi / 4])
    array([0., 1.], dtype=float32)

    """


def tanh(arg0: TensorLike) -> Tensor:
    """Return element-wise hyperbolic tangent value of the tensor object.

    Parameters
    ----------
    arg0 : TensorLike
        Input tensor

    Returns
    -------
    Tensor
        Element-wise hyperbolic tangent value of the tensor.

    Examples
    --------
    >>> xg.tanh([-100, 0, 100])
    array([-1.,  0.,  1.], dtype=float32)

    """
