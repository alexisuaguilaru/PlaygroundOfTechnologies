#include <Python.h>
#include "Class.hpp"

/*
Basic struct used like an interface between 
Python and C/C++ objects
*/
typedef struct 
{
    PyObject_HEAD
    Rectangle* CppObject;
} RectanglePyObject;


/*
Methods in a Python class
*/
static PyMethodDef 
Rectangle_Methods[] = {
    {NULL}
};

/*
Methods for setting and getting attributes of an object
*/
static PyGetSetDef 
Rectangle_GetSetters[] = {
    {NULL}
};


/*
Struct used to define a Python class 
Definition of the type (class)

@note The designator order is important, check https://docs.python.org/3/c-api/typeobj.html#examples for order
*/
static PyTypeObject 
RectanglePyType = {
    PyVarObject_HEAD_INIT(NULL,0) // Init a type object in Python
    .tp_name = "ClassModule.Rectangle", // access to the class by: from ClassModule import Rectangle
    
    .tp_basicsize = sizeof(RectanglePyObject), // Size in bytes of instances of the type
    /*
    0 means that type creates fixed-length instances (not contains Python list or strings, or dynamic memory)
    != 0 means that type creates variable-length instances (uses dynamic memory to grow)
    */
   .tp_itemsize = 0, 
   
   .tp_flags = Py_TPFLAGS_DEFAULT, // Base behavior of a class in Python
   
   .tp_doc = PyDoc_STR("Class for defining a Rectangle"),

    .tp_methods = Rectangle_Methods,
    .tp_getset = Rectangle_GetSetters,
};


// Structs and definition for module functionality

static PyMethodDef 
ClassModule_Methods[] = {
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef 
ClassModule = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "RectangleModule",
    .m_doc = "Rectangle Class in C++",
    .m_methods = ClassModule_Methods,
};

PyMODINIT_FUNC 
PyInit_RectangleModule(void)
{
    PyObject* Module;

    if (PyType_Ready(&RectanglePyType) < 0)
        return NULL;
    
    Module = PyModule_Create(&ClassModule);
    if (Module == NULL)
        return NULL;
    
    /*
    Code for controlling the reference counting logic when a class is referenced
    */
    Py_INCREF(&RectanglePyType);
    // This allows use the new type as Module.Name
    if (PyModule_AddObject(Module,"Rectangle",(PyObject*)&RectanglePyType) < 0)
    {
        Py_DECREF(&RectanglePyType);
        Py_DECREF(Module);
        return NULL;
    }
    
    return Module; 
};