from numbers import Number

class Rectangle:
    """
    Class to implement with Python/C API
    """

    def __init__(
            self,
            Name: str,
            Base: Number,
            Height: Number,
        ):
        self.Name = Name
        self.Base = Base
        self.Height = Height
    
    def __str__(self):
        return f'Rectangle {self.Name} :: {self.Base} x {self.Height}'
    
    def __lt__(self,other):
        return self.Area() < other.Area()
    
    def __eq__(self,other):
        return self.Area() == other.Area()
    
    def Area(self):
        return self.Base * self.Height