import example_02__module

from time import time

if __name__ == '__main__':
    start = time()

    ans = example_02__module.PositionalArgs(1,0.5,'Hello')
    print(ans)

    ans = example_02__module.VaritonalArgs(-1,1,1,2,3,4,5,6)
    print(ans)

    example_02__module.PositionalKeywordsArgs(0,1,a=1,b=0.6,c=0.1)
    example_02__module.PositionalKeywordsArgs(-1,2,a=1,b=0.6)

    ans = example_02__module.KeywordsArgs(a=123,b='cc',c=0.5)
    print(ans)

    end = time()
    
    print(end-start)