## Brief
For this mini-experiment, a minimal example of a Python function was made and annotations were made, as documentation, to be consulted later on the processing taken to write a Python module in C.

## Comments
First, the official Python references and manuals, [Extending Python with C or C++](https://docs.python.org/3/extending/extending.html) and [Python/C API Reference Manual](https://docs.python.org/3/c-api/index.html), were consulted to generate a general idea of how to create a Python module in C. This task was accomplished, so it was proceeded to define the different variables and structs found in [Example.c](./Example.c).

To compile the source code and be used as a module in Python, it was consulted with [DeepSeek](https://chat.deepseek.com/) in search of modern alternatives, where it emerges to use [Setuptools](https://setuptools.pypa.io/en/latest/) which has a simple interface to execute the commands for the compilation of the code. This can be seen in [setup.py](./setup.py) where you only need to reference the name of the struct containing the module init in [Example.c](./Example.c) and the source code itself. For its compilation it is required to execute the following commands:
```bash
pip install setuptools 
python setup.py build_ext --inplace
```

## Results
Finally, [Example.py](./Example.py) shows an example of how the newly compiled module is used and also that it works; thus achieving the goal of this mini experiment.