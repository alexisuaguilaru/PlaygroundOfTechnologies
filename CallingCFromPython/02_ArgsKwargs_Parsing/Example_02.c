#include <Python.h>

/*
Example of a function that only accept positional arguments
*/
static PyObject*
example_02__PositionalArgs(PyObject *self, PyObject *args)
{
    int a;
    float b , part;
    const char *c;

    if (!PyArg_ParseTuple(args, "ifs", &a, &b, &c))
        return NULL;

    part = b*a;

    /* Create a dynamic size buffer*/
    int size_buffer = snprintf(NULL, 0, "%s\t%f", c, part) + 1;
    char *buffer = (char *)malloc(size_buffer);
    
    /* Create a f-string based on a printf-style format */
    snprintf(buffer, size_buffer, "%s\t%f", c, part);
    PyObject *format_result = PyUnicode_FromString(buffer);

    free(buffer);

    return format_result;
}

static PyMethodDef
example_02__methods[] = 
{
    {"PositionalArgs", example_02__PositionalArgs, METH_VARARGS, "Function that only accepted positional arguments. No *args neither **kwargs"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef 
example_02__module =
{
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "example_02__module",
    .m_doc = "Differents type of arguments parsing",
    .m_methods = example_02__methods,
};

PyMODINIT_FUNC 
PyInit_example_02__module(void)
{
    return PyModule_Create(&example_02__module); 
};