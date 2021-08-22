import typing as tp

import numpy as np


class Tensor:
    """Class for N-dimensional tensor object.

    Examples
    --------
    >>> a = xg.Tensor(np.array([[1, 2, 3], [4, 5, 6]]))
    >>> a
    array([[1., 2., 3.],
           [4., 5., 6.]], dtype=float32)
    >>> a.ndim
    2
    >>> a.size
    6
    >>> a.shape
    [2, 3]
    >>> a.strides
    [3, 1]

    """

    def __init__(self, arg0: np.ndarray) -> None:
        """Initialize a tensor object.

        Parameters
        ----------
        arg0 : np.ndarray
            N-dimensional array.
        """

    def __repr__(self) -> str: ...

    def __neg__(self) -> Tensor:
        """Return negation of the tensor object.

        Returns
        -------
        Tensor
            Negated tensor object.
        """

    @property
    def is_view(self) -> bool:
        """Return true if the tensor object is a view of another.

        Returns
        -------
        bool
            True if the tensor object is a view of another otherwise false.
        """

    @property
    def ndim(self) -> int:
        """Return dimensionality of the tensor object.

        Returns
        -------
        int
            Dimensionality of the tensor object.
        """

    @property
    def size(self) -> int:
        """Return number of elements in the tensor object.

        Returns
        -------
        int
            Number of elements in the tensor object.
        """

    @property
    def shape(self) -> tp.List[int]:
        """Return shape of the tensor object

        Returns
        -------
        tp.List[int]
            Shape of the tensor object.
        """

    @property
    def strides(self) -> tp.List[int]:
        """Return strides of the tensor object.

        Returns
        -------
        tp.List[int]
            Stride of the tensor object for each axis.
        """

    @property
    def data(self) -> np.ndarray:
        """Return numpy array of the tensor data.

        Returns
        -------
        np.ndarray
            Numpy array of the tensor data.
        """

    @property
    def grad(self) -> np.ndarray:
        """Return numpy array of the tensor grad.

        Returns
        -------
        np.ndarray
            Numpy array of the tensor grad.

        Raises
        ------
        ValueError
            The object is constant and grad does not exist.
        """

    @property
    def requires_grad(self) -> bool:
        """Return true if the object is a differentiable variable.

        Returns
        -------
        bool
            True if the object is a differentiable variable otherwise false.
        """

    @requires_grad.setter
    def requires_grad(self, flag: bool) -> None: ...

    def backward(self) -> None:
        """Backpropagate gradient from the tensor object.

        Backpropagate gradient of the tensor object using the chain rules of
        differentiation.
        """
