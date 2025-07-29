from setuptools import setup , Extension

setup(
    ext_modules=[
        Extension(
            'RectangleModule',
            sources=['Class.cpp',],
            language='c++',
            extra_compile_args=['-std=c++11'],
        )
    ]
)