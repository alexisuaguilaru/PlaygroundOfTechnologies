from setuptools import setup , Extension

setup(
    ext_modules=[
        Extension(
            'ClassModule',
            sources=['Class.cpp','Class.hpp'],
            language='c++',
        )
    ]
)