import numpy
import pytest

import xgrad


@pytest.fixture(autouse=True)
def add_np(doctest_namespace):
    doctest_namespace["np"] = numpy


@pytest.fixture(autouse=True)
def add_xg(doctest_namespace):
    doctest_namespace["xg"] = xgrad
