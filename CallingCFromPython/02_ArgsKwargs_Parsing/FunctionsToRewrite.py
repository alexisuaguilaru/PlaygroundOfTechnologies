from numbers import Number
from itertools import starmap
from typing import Iterator

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
        base:float=0.5,
        augm:int=1,
        **kwargs:dict[str,Number],
    ) -> None:
    """
    Function that accept positional/default 
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

def AllTypeArgs(
        positional:float,
        *args:list[int],
        default:str='world',
        **kwargs,
    ) -> Iterator:
    """
    Function that accept every type of 
    argument and yield values
    """
    yield positional*sum(args,start=1)
    yield KeywordsArgs(**{'default':default,**kwargs})

if __name__ == '__main__':
    ans = PositionalArgs(1,0.5,'Hello')
    print(ans)

    ans = VaritonalArgs(-1,1,1,2,3,4,5,6)
    print(ans)

    PositionalKeywordsArgs(a=1,b=0.6,c=0.1)
    PositionalKeywordsArgs(base=-1,augm=2,a=1,b=0.6)

    ans = KeywordsArgs(a=123,b='cc',c=0.5)
    print(ans)

    for ans in AllTypeArgs(0.5,1,1,12,3,54,a='0',b='1',c='2'):
        print(ans)