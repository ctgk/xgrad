import numpy as np
import pytest

import xgrad as xg


@pytest.fixture(params=[
    {
        'op': xg.cos,
        'input': np.array([[0, 1, -1], [np.pi / 2, 3, -np.pi]]),
        'forward': np.cos([[0, 1, -1], [np.pi / 2, 3, -np.pi]]),
        'backward': -np.sin([[0, 1, -1], [np.pi / 2, 3, -np.pi]]),
    },
    {
        'op': xg.cosh,
        'input': np.linspace(-10, 10, 10),
        'forward': np.cosh(np.linspace(-10, 10, 10)),
        'backward': np.sinh(np.linspace(-10, 10, 10)),
    },
    {
        'op': xg.exp,
        'input': np.array([-1, -0.5, 2, 10]),
        'forward': np.exp([-1, -0.5, 2, 10]),
        'backward': np.exp([-1, -0.5, 2, 10]),
    },
    {
        'op': xg.log,
        'input': np.linspace(0.1, 5, 10),
        'forward': np.log(np.linspace(0.1, 5, 10)),
        'backward': 1 / np.linspace(0.1, 5, 10),
    },
    {
        'op': xg.negate,
        'input': np.array([-1, 2, -3]),
        'forward': np.array([1, -2, 3]),
        'backward': np.array([-1, -1, -1]),
    },
    {
        'op': xg.sin,
        'input': np.array([[0, 1, -1], [np.pi / 2, 3, -np.pi]]),
        'forward': np.sin([[0, 1, -1], [np.pi / 2, 3, -np.pi]]),
        'backward': np.cos([[0, 1, -1], [np.pi / 2, 3, -np.pi]]),
    },
    {
        'op': xg.sinh,
        'input': np.linspace(-10, 10, 10),
        'forward': np.sinh(np.linspace(-10, 10, 10)),
        'backward': np.cosh(np.linspace(-10, 10, 10)),
    },
    {
        'op': xg.square,
        'input': np.array([-1, 2, -3]),
        'forward': np.array([1, 4, 9]),
        'backward': np.array([-2, 4, -6]),
    },
    {
        'op': xg.tan,
        'input': np.array(np.pi * 0.25),
        'forward': 1,
        'backward': 2,
    },
    {
        'op': xg.tanh,
        'input': np.array(0),
        'forward': 0,
        'backward': 1,
    },
])
def parameter(request):
    return request.param


def test_forward(parameter):
    actual = parameter['op'](parameter['input'])
    assert np.allclose(actual.data, parameter['forward'], rtol=0, atol=1e-3)


def test_backward(parameter):
    a = xg.Tensor(parameter['input'])
    a.requires_grad = True
    parameter['op'](a).backward()
    actual = a.grad
    assert np.allclose(actual, parameter['backward'], rtol=0, atol=1e-3)


if __name__ == '__main__':
    pytest.main([__file__])
