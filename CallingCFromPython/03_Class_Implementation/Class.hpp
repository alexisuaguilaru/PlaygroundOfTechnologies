#include <iostream>
#include <Python.h>

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