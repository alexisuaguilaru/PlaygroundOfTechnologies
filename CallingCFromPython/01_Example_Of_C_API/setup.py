from setuptools import setup , Extension

"""
Configuration for compile source C code into 
a Python module
"""
setup(
    ext_modules=[
        Extension(
            'example_module',  # Module name in Python. It must be the same in PyInit_
            sources=['Example.c'],  # Source file
        )
    ]
)