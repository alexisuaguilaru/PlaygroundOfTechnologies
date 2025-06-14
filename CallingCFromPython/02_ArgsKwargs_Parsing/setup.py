from setuptools import setup , Extension

setup(
    ext_modules=[
        Extension(
            'example_02__module',
            sources=['Example_02.c'],
        )
    ]
)