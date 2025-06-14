import example_02__module

from time import time

if __name__ == '__main__':
    start = time()

    ans = example_02__module.PositionalArgs(1,0.5,'Hello')
    print(ans)

    ans = example_02__module.VaritonalArgs(-1,1,1,2,3,4,5,6)
    print(ans)

    end = time()
    
    print(end-start)