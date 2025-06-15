from numbers import Number
from itertools import starmap
from typing import Iterator

from time import time

def PositionalArgs(
        a:int,
        b:float,
        c:str,
    ) -> str:
    """
    Function that only accepted positional 
    arguments. No *args neither **kwargs
    """
    part = a*b

    return f'{c}\t{part}'

def VaritonalArgs(
        init:float,
        *args:list[int]
    ) -> float:
    """
    Function that accept a fixed and 
    varitional positional arguments. No 
    **kwargs 
    """
    sum = init
    for value in args:
        sum += value

    return sum

def PositionalKeywordsArgs(
        base:float,
        augm:int,
        **kwargs:dict[str,Number],
    ) -> None:
    """
    Function that accept positional 
    and keywords arguments. No *args
    """
    for _key , _value in kwargs.items():
        print(_key,base + _value*augm)

def KeywordsArgs(
        **kwargs,
    ) -> str:
    """
    Function that only accept keyword 
    arguments. No *args
    """
    string_result = lambda _key , _value : f'{_key} :: {_value}'
    return '\t\t'.join(starmap(string_result,kwargs.items()))

if __name__ == '__main__':
    start = time()

    ans = PositionalArgs(1,0.5,'Hello')
    print(ans)

    ans = VaritonalArgs(-1,1,1,2,3,4,5,6)
    print(ans)

    PositionalKeywordsArgs(0,1,a=1,b=0.6,c=0.1)
    PositionalKeywordsArgs(-1,2,a=1,b=0.6)

    end = time()

    ans = KeywordsArgs(a=123,b='cc',c=0.5)
    print(ans)

    print(end-start)