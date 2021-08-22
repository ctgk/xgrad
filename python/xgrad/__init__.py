"""XGrad package.

Gradient computation library
"""

from xgrad._version import __version__  # noqa: F401
from xgrad._xgrad_cpp import Tensor
from xgrad._xgrad_cpp import (  # noqa: I100
    cos, cosh, exp, log, negate, sin, sinh, square, tan, tanh,
)


Tensor.__repr__ = lambda self: repr(self.data)


_classes = [
    Tensor,
]


for _cls in _classes:
    _cls.__module__ = __name__


_functions = [
    cos, cosh, exp, log, negate, sin, sinh, square, tan, tanh,
]


__all__ = [
    _cls.__name__ for _cls in _classes
] + [
    _func.__name__ for _func in _functions
]


del _classes
del _cls
del _functions
