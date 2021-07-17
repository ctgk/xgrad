import numpy as np
import pytest

import xgrad as xg


def test_repr():
    a = np.random.rand(2, 5).astype(np.float32)
    b = xg.Tensor(a)
    assert repr(a) == repr(b)


def test_data():
    a = xg.Tensor(np.array([[1, 2], [3, 4]]))
    assert np.allclose(np.array([[1, 2], [3, 4]]), a.data)
    a.data[0] = np.array([5, 6])
    assert np.allclose(np.array([[5, 6], [3, 4]]), a.data)


def test_grad():
    a = xg.Tensor(np.random.rand(3, 2))
    with pytest.raises(ValueError):
        a.grad
    a.requires_grad = True
    assert np.allclose(a.grad, 1)
    a.grad[:, 1] = np.array([2, 3, -1])
    assert np.allclose(a.grad, np.array([[1, 2], [1, 3], [1, -1]]))
    a.requires_grad = False
    with pytest.raises(ValueError):
        a.grad


def test_requires_grad():
    a = xg.Tensor(np.random.rand(3, 2, 4))
    assert a.requires_grad is False
    a.requires_grad = True
    assert a.requires_grad is True
    a.requires_grad = False
    assert a.requires_grad is False


if __name__ == '__main__':
    pytest.main([__file__])
