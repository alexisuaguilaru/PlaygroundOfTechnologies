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

/*
Function that accept a fixed and varitional positional arguments. No **kwargs 

@note Py_ParseTuple is not required because accessing arguments by their indexes
*/
static PyObject*
example_02__VaritonalArgs(PyObject *self, PyObject *args)
{
    int sum = 0;

    /* 
    args is a sequence of values itself that is indexable
    So it is like a Python tuple 
    */
    Py_ssize_t size_varitional_args = PyTuple_Size(args);
    for (Py_ssize_t index = 0; index < size_varitional_args; index++)
    {
        /* Accessing arguments by index return a item */
        PyObject* item =  PyTuple_GetItem(args, index);
        if (index == 0)
        {
            /* The item is recast into a C type */
            sum = PyLong_AsInt(item); /* Converts a Python int into a C int */
        }
        else
        {
            sum += PyLong_AsInt(item);
        }
        
    }

    /* 
    Py_BuildValue could be used instead of use PyLong_FromLong
        return Py_BuildValue("i",sum);
    
    Return a Python int object 
    */
    return PyLong_FromLong(sum);
}

/*
Function that accept positional and keywords arguments. No *args
*/
static PyObject*
example_02__PositionalKeywordsArgs(PyObject *self, PyObject *args, PyObject *kwargs)
{
    float base = 0;
    int augm = 0;
    
    if (!PyArg_ParseTuple(args, "fi", &base, &augm))
        return NULL;

    if (kwargs)
    {
        PyObject *key , *value;
        Py_ssize_t index_item = 0;

        while (PyDict_Next(kwargs, &index_item, &key, &value))
        {
            PyObject* numeric_value = PyNumber_Float(value);
            float _value = PyFloat_AsDouble(numeric_value);
            printf("%s %f\n", PyUnicode_AsUTF8(key), base+augm*_value);
        }
    }

    Py_RETURN_NONE;
}

static PyMethodDef
example_02__methods[] = 
{
    {"PositionalArgs", example_02__PositionalArgs, METH_VARARGS, "Function that only accepted positional arguments. No *args neither **kwargs"},
    {"VaritonalArgs", example_02__VaritonalArgs, METH_VARARGS, "Function that accepted fixed and varitional positional arguments. No **kwargs"},
    {"PositionalKeywordsArgs", (PyCFunction)example_02__PositionalKeywordsArgs, METH_VARARGS | METH_KEYWORDS, "Function that accept positional and keywords arguments. No *args"},
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