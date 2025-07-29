#include <Python.h>
#include <iostream>

/*
@brief Class being implemented in with Python/C API

* Python definition in ClassToImplement.py
*/
class Rectangle
{
    public:
        std::string Name;
        double Base , Height;

        /*__init__ class method*/
        Rectangle(const std::string& name,double base,double height) : Name(name), Base(base), Height(height) {}
        double Area()
        {
            return Base*Height;
        }
};

/*
Basic struct used like an interface between 
Python and C/C++ objects
*/
typedef struct 
{
    PyObject_HEAD
    Rectangle* CppObject;
} RectanglePyObject;