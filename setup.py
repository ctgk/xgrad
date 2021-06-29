from setuptools import setup


develop_requires = [
    'pre-commit',
]

setup(
    extras_require={
        'develop': develop_requires,
    },
)
