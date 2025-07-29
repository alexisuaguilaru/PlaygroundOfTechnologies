from ClassToImplement import Rectangle
import RectangleModule

if __name__ == '__main__':
    rect_A_cpp = RectangleModule.Rectangle('A',2,2)
    rect_B_cpp = RectangleModule.Rectangle('B',3.0,4.0)

    rect_A = Rectangle('A',2,2)
    rect_B = Rectangle('B',3.0,4.0)

    print(' Rectangles Python '.center(15,'='),rect_A,rect_B,sep='\n')
    print(f'Is A == B :: {rect_A == rect_B}')
    print(f'Is A < B :: {rect_A < rect_B}')

    print(' Rectangles Cpp '.center(15,'='),rect_A_cpp,rect_B_cpp,sep='\n')
    print(f'Is A == B :: {rect_A_cpp == rect_B_cpp}')
    print(f'Is A < B :: {rect_A_cpp < rect_B_cpp}')