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
        Rectangle(
            const std::string& name,
            double base,
            double height
        );

        /* __del__ class method*/        
        ~Rectangle();

        /* magical class methods */
        std::string Rectangle_str() const;
        bool Rectangle_lt(const Rectangle& other) const;
        bool Rectangle_eq(const Rectangle& other) const;

        /* other class methods */
        double Rectangle_Area() const;
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


// Methods in a Python class //

/*
@brief __del__ class method.
*/
static void Rectangle_dealloc(RectanglePyObject* self)
{
    delete self->CppObject;
    Py_TYPE(self)->tp_free((PyObject*)self);
}

/*
@brief __new__ class method.
It is like:
   
    def __new__(cls, *args, **kwargs): ...
*/
static PyObject* 
Rectangle_new(PyTypeObject* Type,PyObject* args,PyObject* kwargs)
{   
    // Create a new object and assign it memory
    RectanglePyObject* self;
    self = (RectanglePyObject*)Type->tp_alloc(Type, 0);

    if (self != NULL) {
        // Create a null reference to a C++ object
        self->CppObject = nullptr;
    }
    return (PyObject*)self;
}

/*
@brief __init__ class method
*/
static int 
Rectangle_init(RectanglePyObject* self,PyObject* args,PyObject* kwargs)
{
    double Base = 0, Height = 0;
    const char* Name = "";
    static char* kwlist[] = {"Name", "Base", "Height", NULL};

    if (!PyArg_ParseTupleAndKeywords(args,kwargs,"sdd", kwlist, &Name, &Base, &Height)) {
        return -1;
    }

    self->CppObject = new Rectangle(std::string(Name),Base,Height);
    return 0;
}

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

   .tp_dealloc = (destructor)Rectangle_dealloc,
   
   .tp_flags = Py_TPFLAGS_DEFAULT, // Base behavior of a class in Python
   
   .tp_doc = PyDoc_STR("Class for defining a Rectangle"),

    .tp_methods = Rectangle_Methods,
    .tp_getset = Rectangle_GetSetters,

    .tp_init = (initproc)Rectangle_init,
    .tp_new = Rectangle_new,
};


// Structs and definition for module functionality //

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