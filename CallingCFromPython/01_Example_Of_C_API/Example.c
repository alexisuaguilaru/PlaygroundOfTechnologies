#include <Python.h>

/* 
* @brief Interface for calling the function hello_world written in C. 
* 
* @param self Points to the module
* @param args Points to the Python arguments 
*/
static PyObject* 
example_hello_world(PyObject *self, PyObject *args) 
{
    const char *print_str;
    // int status;

    if (!PyArg_ParseTuple(args, "s", &print_str))
        return NULL;

    printf("%s\n",print_str);

    return Py_None;
}

/*
* @brief Definition of methods/functions table of the module
* 
* @note Contains each function available in the Python module 
*/
static PyMethodDef 
example_methods[] = {
    /* Declaration of a module method. Name, address funtion, type of calling convention, docstring*/
    {"hello_world", example_hello_world, METH_VARARGS, "Print string conteted in print_str"}, 
    {NULL, NULL, 0, NULL} /* Sentinel. Final */
};

/*
* @brief Definition of module itself. Its name, docstring, table of methos
* 
* @note PyModuleDef_HEAD_INIT it is mandatory for creation of a module
*/
static struct PyModuleDef 
example_module = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "example_module",
    .m_doc = "Docs for Example Module",
    .m_methods = example_methods,
};

/*
* @brief Initialization function for module, it is like the bridge between C and Python
*/
PyMODINIT_FUNC 
PyInit_example_module(void)
{
    return PyModule_Create(&example_module); 
};