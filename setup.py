"""Setup xgrad package."""

from glob import glob

from pybind11.setup_helpers import (
    build_ext, ParallelCompile, Pybind11Extension,
)
from setuptools import setup


ParallelCompile('NPY_NUM_BUILD_JOBS').install()


extension_modules = [
    Pybind11Extension(
        name='xgrad._xgrad_cpp',
        sources=sorted(
            glob('cpp/src/**/*.cpp', recursive=True)
            + glob('python/src/*.cpp', recursive=True),
        ),
        include_dirs=['cpp/include', 'cpp/src/', 'python/src'],  # -I
        define_macros=[],  # -D<string>=<string>
        undef_macros=[],  # [string] -D<string>
        library_dirs=[],  # [string] -L<string>
        runtime_library_dirs=[],  # [string] -rpath=<string>
        extra_objects=[],
        extra_compile_args=[],
        extra_link_args=[],
    ),
]


setup(
    ext_modules=extension_modules,
    cmdclass={'build_ext': build_ext},
)
