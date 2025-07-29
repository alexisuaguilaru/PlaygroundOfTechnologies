#include <Python.h>
#include <iostream>

/*
@brief Class being implemented in with Python/C API

* Python definition in ClassToImplement.py
*/
class Rectangle
{
    private:
        std::string Name;
        double Base , Height;

    public:
        /*__init__ class method*/
        Rectangle(std::string name,double base,double height) : Name(name), Base(base), Height(height) {}
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