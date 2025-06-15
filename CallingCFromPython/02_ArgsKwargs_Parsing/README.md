## Brief
For this experiment is intended to reimplement the functions of [FunctionsToRewrite.py](./FunctionsToRewrite.py) in C, this with the aim of learning how to use the parsers of positional, variable number of and keywords arguments.

## Comment
First consult the following resources for a better understanding of positional argument and keyword parsing, and the return of python objects [Parsing arguments and building values](https://docs.python.org/3/c-api/arg.html), [Concrete Objects Layer](https://docs.python.org/3/c-api/concrete.html) and [Abstract Objects Layer](https://docs.python.org/3/c-api/abstract.html).

By consulting these resources it was possible to give an outline on how the Python functions in [Example_02.c](./Example_02.c) could be re-implemented to Python-compatible C functions. Where the major insight is how to handle the cases when arguments are fixed and dynamic, the former being the easiest; during the review and from the resources, it was also noted how to implement the parsing of optional arguments by making use of "|" and `PyArg_ParseTupleAndKeywords`.

## Results
Finally, [Example_02](./Example_02.py) shows the use and calling of the re-implemented functions. When compared to the functions in [FunctionsToRewrite.py](./FunctionsToRewrite.py) they take similar execution times, but it makes sense when considering the simplicity of the functions.