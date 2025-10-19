################################################################################
#
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
################################################################################
from dataclasses import dataclass
import inspect
from typing import List, Optional, Union, Callable
from little_kernel.core.type_system import LLType

INTERNAL_ATTR = [
    "__builtin__",
    "__eval_return_type__",
    "__eval_arg_type__",
    "__const_func__",
    "__const_type_func__",
    "__inline_func__",
    "__codegen_func__",
    "__original_py_func__",
]


def builtin(eval_return_type: Union[LLType, Callable],
            codegen_func: Callable,
            eval_arg_type: Optional[Callable] = None):

    def _builtin(func):

        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        wrapper.__module__ = func.__module__
        for attr in INTERNAL_ATTR:
            if hasattr(func, attr):
                setattr(wrapper, attr, getattr(func, attr))
        if not hasattr(wrapper, "__original_py_func__"):
            wrapper.__original_py_func__ = func
        wrapper.__builtin__ = True
        if isinstance(eval_return_type, LLType):
            eval_return_type_func = lambda *_, **kwargs: eval_return_type
        else:
            eval_return_type_func = eval_return_type
        if eval_arg_type is not None:
            wrapper.__eval_arg_type__ = eval_arg_type
        wrapper.__codegen_func__ = codegen_func
        assert isinstance(eval_return_type_func, Callable)
        wrapper.__eval_return_type__ = eval_return_type_func
        return wrapper

    return _builtin


def const_func(func):

    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    wrapper.__module__ = func.__module__
    for attr in INTERNAL_ATTR:
        if hasattr(func, attr):
            setattr(wrapper, attr, getattr(func, attr))
    if not hasattr(wrapper, "__original_py_func__"):
        wrapper.__original_py_func__ = func
    wrapper.__const_func__ = True
    return wrapper


def const_type_func(func):

    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    wrapper.__module__ = func.__module__
    for attr in INTERNAL_ATTR:
        if hasattr(func, attr):
            setattr(wrapper, attr, getattr(func, attr))
    if not hasattr(wrapper, "__original_py_func__"):
        wrapper.__original_py_func__ = func
    wrapper.__const_type_func__ = True
    return wrapper


def inline_func(func):

    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    wrapper.__module__ = func.__module__
    for attr in INTERNAL_ATTR:
        if hasattr(func, attr):
            setattr(wrapper, attr, getattr(func, attr))
    if not hasattr(wrapper, "__original_py_func__"):
        wrapper.__original_py_func__ = func
    wrapper.__inline_func__ = True
    return wrapper


def builtin_class(cls):
    cls.__builtin__ = True
    return cls


@dataclass
class Builtin:
    body: str
    includes: List[str]
    return_val: str
