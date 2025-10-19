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

import ast
import inspect
from .internal import __LITTLE_KERNEL_ENTRY__


class LLKernel:

    def __init__(self, py_func, backend, is_entry):
        self.py_func = py_func
        self.ctx = py_func.__globals__
        self.backend = backend

        self.is_entry = is_entry
        if is_entry:
            self.ctx[__LITTLE_KERNEL_ENTRY__] = self.py_func

    def compile(self, passes, codegen_func, need_header=True):
        tree = self.lower(passes)
        return codegen_func(tree, self.ctx, emit_header=need_header)

    def lower(self, passes):
        source = inspect.getsource(self.py_func)

        tree = ast.parse(source)
        for p in passes:
            tree = p(tree, self.ctx)
        return tree

    def __call__(self, *args, **kwargs):
        self.ctx[__LITTLE_KERNEL_ENTRY__] = self.py_func


def ll_kernel(backend="cuda", is_entry=False):

    def _compile_helper(func):
        compiled_func = LLKernel(func, backend, is_entry)

        compiled_func.__name__ = func.__name__
        compiled_func.__doc__ = func.__doc__
        compiled_func.__module__ = func.__module__
        return compiled_func

    return _compile_helper
