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

from .constfold import *
from .inline import *
from .mem_analysis import *
from .utils.add_parent_reference import *
from .insert_mem_alloc import *
from .pass_base import *
from .flatten_empty import flatten_empty
from .dump import dump_ast
import os

PASS_DEBUG = os.environ.get("PASS_DEBUG", "").split(",")
if "" in PASS_DEBUG:
    PASS_DEBUG.remove("")
PASS_DEBUG_ALL = "all" in PASS_DEBUG


def empty_pass(tree, ctx, *args, **kwargs):
    return tree


def debug_wrapper(pass_func, pass_name):

    def wrapper_func(tree, ctx, *args, **kwargs):
        tree = pass_func(tree, ctx, *args, **kwargs)
        if pass_name in PASS_DEBUG or PASS_DEBUG_ALL:
            dump_ast(f"after {pass_name}")(tree, ctx, *args, **kwargs)
        return tree

    return wrapper_func


PASSES = {
    "cuda": [
        dump_ast("initial") if len(PASS_DEBUG) > 0 else empty_pass,
        debug_wrapper(const_fold, "const_fold"),
        debug_wrapper(inline, "inline"),
        debug_wrapper(flatten_empty, "flatten_empty"),
        debug_wrapper(insert_mem_alloc, "insert_mem_alloc"),
    ]
}
