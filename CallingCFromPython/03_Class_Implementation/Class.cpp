#include <Python.h>
#include "Class.hpp"

/*
Basic struct used like a interfance between 
Python and C/C++ objects
*/
typedef struct 
{
    PyObject_HEAD
    Rectangle* CppObject;
} RectanglePyObject;


// Structs and definition for module functionality

static PyMethodDef 
ClassModule_Methods[] = {
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef 
ClassModule = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "RectangleModule",
    .m_doc = "Rectanle Class in C++",
    .m_methods = ClassModule_Methods,
};

PyMODINIT_FUNC 
PyInit_ClassModule(void)
{
    return PyModule_Create(&ClassModule); 
};