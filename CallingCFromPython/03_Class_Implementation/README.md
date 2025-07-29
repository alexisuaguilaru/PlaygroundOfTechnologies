## Brief
For this experiment is intended to implement the simple class in [ClassToImplement.py](./ClassToImplement.py) in C/C++ using the Python/C API, this with the aim of learn how to call and use a class written in C/C++ from Python.

## Comment
For Python/C API, create a new type is equivalent to create a new class in pure Python. So, [Defining Extension Types: Tutorial](https://docs.python.org/3/extending/newtypes_tutorial.html) was consulted to give an idea about how define a type (class) and consulting other resources it was possible to reimplement the class in [ClassToImplement.py](./ClassToImplement.py) as a new type in [Class.cpp](./Class.cpp).

## Results
Finally, [Example_03](./Example_03.py) shows the use and functionality of the new type in action with a comparison with the class implemented in Python, both with same results.