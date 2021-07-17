"""XGrad package.

Gradient computation library
"""

from xgrad._version import __version__  # noqa: F401
from xgrad._xgrad_cpp import Tensor


Tensor.__repr__ = lambda self: repr(self.data)


_classes = [
    Tensor,
]


for _cls in _classes:
    _cls.__module__ = __name__


__all__ = [_cls.__name__ for _cls in _classes]


del _classes
del _cls
