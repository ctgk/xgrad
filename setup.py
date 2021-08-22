"""Setup xgrad package."""

from glob import glob
import os
import typing as tp

from pybind11.setup_helpers import (
    build_ext, ParallelCompile, Pybind11Extension,
)
from setuptools import setup


ParallelCompile('NPY_NUM_BUILD_JOBS').install()


def _get_docstring(lines: tp.List[str]) -> str:
    indent_length: int = len(lines[0]) - len(lines[0].lstrip(' '))
    start_position = lines[0].find('"""') + 3
    docstring = lines[0][start_position:]
    for ll in lines[1:]:
        if '"""' in ll:
            break
        docstring = docstring + (ll if ll == '\n' else ll[indent_length:])
    docstring = repr(docstring)[1:-1]
    docstring = '"' + (docstring) + '"'
    return docstring


def _get_docstring_starting_lines(lines: tp.List[str]) -> tp.List[int]:
    line_numbers: tp.List[int] = []
    for ii, ll in enumerate(lines):
        if len(ll.lstrip(' ')) > 5 and '"""' in ll:
            line_numbers.append(ii)
    return line_numbers


def _declaration_to_macro(line: str, namespaces: tp.List[str]) -> str:
    if 'class' in line:
        name = line.lstrip(' ').split(' ')[1].split(':')[0].upper()
    elif 'def' in line:
        name = line.lstrip(' ').split(' ')[1].split('(')[0]
        if name.count('__') == 2:
            name = name.split('__')[1]
        name = name.upper()
    else:
        raise ValueError(f'Unsupported declaration: {line}')
    return f'{"_".join(map(str.upper, namespaces))}_{name}_DOCSTRING'


def _make_macro_docstring_pair(
    filepath_relative_to_root: tp.List[str],
    declaration: str,
    lines: tp.List[str],
) -> tp.Tuple[str, str]:
    hierarchies = filepath_relative_to_root.split('/')
    namespaces = [h for h in hierarchies[:-1] if not h.startswith('_')]
    if declaration.startswith('    '):
        namespaces.append(hierarchies[-1].split('.')[0].lstrip('_'))

    declaration = _declaration_to_macro(declaration, namespaces)
    docstring = _get_docstring(lines)
    return declaration, docstring


def _macro_docstring_pairs(
    filepath_relative_to_root: str,
) -> tp.List[tp.Tuple[str, str]]:
    filepath = os.path.join(
        os.path.abspath(os.path.dirname(__file__)),
        'python',
        filepath_relative_to_root,
    )
    with open(filepath, 'r') as f:
        lines: tp.List[str] = f.readlines()
    line_numbers = _get_docstring_starting_lines(lines)
    return [
        _make_macro_docstring_pair(
            filepath_relative_to_root, lines[ii - 1], lines[ii:],
        ) for ii in line_numbers
    ]


extension_modules = [
    Pybind11Extension(
        name='xgrad._xgrad_cpp',
        sources=sorted(
            glob('cpp/src/**/*.cpp', recursive=True)
            + glob('python/src/*.cpp', recursive=True),
        ),
        include_dirs=['cpp/include', 'cpp/src/', 'python/src'],  # -I
        define_macros=[
            *_macro_docstring_pairs('xgrad/_tensor.pyi'),
            *_macro_docstring_pairs('xgrad/_math.pyi'),
        ],  # (str1, str2) -D<str1>=<str2>
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
