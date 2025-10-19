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
from typing import Dict, Any, Set, List, Callable, Optional

from .pass_base import CompleteASTMutator
from ..type_system import LLType
from .utils.resolve_attribute import recursive_resolve_attribute
from .utils.infer_type import infer_type, TypeInferencer


class ConstantFolder(CompleteASTMutator):

    def __init__(self, ctx=None, all_types=None):
        super().__init__()
        self.ctx = ctx if ctx is not None else {}
        self.defines = {}
        self.is_rvalue = False
        self.is_rvalue_stack = [False]
        # Map AST comparison operators to their corresponding Python operations
        self._op_mapping = {
            ast.Eq: lambda a, b: a == b,
            ast.NotEq: lambda a, b: a != b,
            ast.Lt: lambda a, b: a < b,
            ast.LtE: lambda a, b: a <= b,
            ast.Gt: lambda a, b: a > b,
            ast.GtE: lambda a, b: a >= b,
            ast.Is: lambda a, b: a is b,
            ast.IsNot: lambda a, b: a is not b,
            ast.In: lambda a, b: a in b,
            ast.NotIn: lambda a, b: a not in b,
        }
        self._op_handlers: Dict[type[ast.operator], Callable[
            [Any, Any], Any]] = {
                ast.Add: lambda a, b: a + b,  # a + b
                ast.Sub: lambda a, b: a - b,  # a - b
                ast.Mult: lambda a, b: a * b,  # a * b
                ast.Div: lambda a, b: a / b,  # a / b
                ast.FloorDiv: lambda a, b: a // b,  # a // b
                ast.Mod: lambda a, b: a % b,  # a % b
                ast.Pow: lambda a, b: a**b,  # a **b
                ast.LShift: lambda a, b: a << b,  # a << b
                ast.RShift: lambda a, b: a >> b,  # a >> b
                ast.BitOr: lambda a, b: a | b,  # a | b
                ast.BitXor: lambda a, b: a ^ b,  # a ^ b
                ast.BitAnd: lambda a, b: a & b,  # a & b
                ast.MatMult:
                lambda a, b: a @ b,  # a @ b (matrix multiplication)
            }
        self._bool_handlers: Dict[type[ast.operator],
                                  Callable[[Any, Any], bool]] = {
                                      ast.And: lambda *values: all(values),
                                      ast.Or: lambda *values: any(values),
                                  }
        self.all_types = all_types if all_types is not None else {}

    def switch_on_use(self):
        self.is_rvalue_stack.append(self.is_rvalue)
        self.is_rvalue = True

    def switch_off_use(self):
        assert len(self.is_rvalue_stack) > 0
        self.is_rvalue = self.is_rvalue_stack.pop()

    def visit_Compare(self, node: ast.Compare) -> ast.AST:
        """
        Fold Compare nodes into Constants if all operands are Constants.
        Example: 3 < 5 -> Constant(value=True)
        Example: 2 + 2 == 4 (after folding) -> Constant(value=True)
        """
        self.switch_on_use()
        # First, recursively process child nodes (in case they contain foldable expressions)
        processed_left = self.visit(node.left)
        processed_comparators = [self.visit(comp) for comp in node.comparators]
        processed_ops = node.ops  # Operators are not AST nodes to process
        self.switch_off_use()

        # Check if all operands are Constants after processing
        if not isinstance(processed_left, ast.Constant):
            # Left operand is not a constant → return original Compare (with processed children)
            return ast.Compare(left=processed_left,
                               ops=processed_ops,
                               comparators=processed_comparators)

        for comp in processed_comparators:
            if not isinstance(comp, ast.Constant):
                # At least one comparator is not a constant → return original Compare
                return ast.Compare(left=processed_left,
                                   ops=processed_ops,
                                   comparators=processed_comparators)

        # Extract constant values
        left_val = processed_left.value
        comparator_vals = [comp.value for comp in processed_comparators]

        # Evaluate chained comparisons (e.g., a < b < c → (a < b) and (b < c))
        try:
            result = True
            current_val = left_val
            for op, comp_val in zip(processed_ops, comparator_vals):
                # Get the operation function (e.g., Lt → lambda a,b: a < b)
                op_type = type(op)
                op_func = self._op_mapping.get(op_type)
                if not op_func:
                    # Unknown operator → cannot fold
                    raise ValueError(
                        f"Unsupported operator: {op_type.__name__}")

                # Evaluate current comparison step
                step_result = op_func(current_val, comp_val)
                if not step_result:
                    # Short-circuit: if any step fails, overall result is False
                    result = False
                    break

                # For chained comparisons, next step uses current comparator as left value
                current_val = comp_val

        except Exception as e:
            # Handle errors (e.g., incompatible types like 5 < "string")
            print(f"Warning: Could not fold comparison: {e}")
            return ast.Compare(left=processed_left,
                               ops=processed_ops,
                               comparators=processed_comparators)

        # Return folded result as a Constant node
        folded_constant = ast.Constant(value=result)
        # Preserve original location info for consistency
        ast.copy_location(folded_constant, node)

        return folded_constant

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        """
        Fold BinOp nodes into Constants if both operands are Constants.
        Example: 2 + 3 → Constant(value=5)
        Example: (4 * 5) - 6 → Constant(value=14)
        """
        # First, recursively process left and right operands (handles nested BinOps)
        self.switch_on_use()
        processed_left = self.visit(node.left)
        processed_right = self.visit(node.right)
        self.switch_off_use()

        # Check if both operands are Constants after processing
        if not (isinstance(processed_left, ast.Constant)
                and isinstance(processed_right, ast.Constant)):
            # At least one operand is not a constant → return original BinOp (with processed children)
            return ast.BinOp(left=processed_left,
                             op=node.op,
                             right=processed_right)

        # Extract constant values
        left_val = processed_left.value
        right_val = processed_right.value
        op_type = type(node.op)

        # Evaluate the binary operation
        try:
            # Get the handler function for this operator (e.g., Add → lambda a,b: a+b)
            op_handler = self._op_handlers.get(op_type)
            if not op_handler:
                raise ValueError(f"Unsupported operator: {op_type.__name__}")

            # Compute the result
            result = op_handler(left_val, right_val)

        except Exception as e:
            # Handle errors (e.g., division by zero, incompatible types like "a" + 5)
            print(f"Warning: Could not fold BinOp {ast.unparse(node)}: {e}")
            return ast.BinOp(left=processed_left,
                             op=node.op,
                             right=processed_right)

        # Return the folded result as a Constant node
        folded_constant = ast.Constant(value=result)
        # Preserve original node's location (line/column numbers)
        ast.copy_location(folded_constant, node)
        return folded_constant

    def visit_BoolOp(self, node: ast.BoolOp) -> ast.AST:
        """
        Fold BoolOp nodes into Constants if both operands are Constants.
        Example: True and False → Constant(value=False)
        """
        # First, recursively process left and right operands (handles nested BinOps)
        self.switch_on_use()
        processed_values = [self.visit(v) for v in node.values]
        self.switch_off_use()

        # Check if all operands are Constants after processing
        for processed_v in processed_values:
            if not isinstance(processed_v, ast.Constant):
                # At least one operand is not a constant → return original BoolOp (with processed children)
                return ast.BoolOp(op=node.op, values=processed_values)

        # Extract constant values
        const_values = [v.value for v in processed_values]
        op_type = type(node.op)

        # Evaluate the handler operation
        try:
            # Get the handler function for this operator
            op_handler = self._bool_handlers.get(op_type)
            if not op_handler:
                raise ValueError(f"Unsupported operator: {op_type.__name__}")

            # Compute the result
            result = op_handler(*const_values)

        except Exception as e:
            # Handle errors (e.g., division by zero, incompatible types like "a" + 5)
            print(f"Warning: Could not fold BoolOp {ast.unparse(node)}: {e}")
            return ast.BoolOp(op=node.op, values=processed_values)

        # Return the folded result as a Constant node
        folded_constant = ast.Constant(value=result)
        # Preserve original node's location (line/column numbers)
        ast.copy_location(folded_constant, node)
        return folded_constant

    def visit_IfExp(self, node: ast.IfExp) -> ast.AST:
        self.switch_on_use()
        new_test = self.visit(node.test)
        if isinstance(new_test, ast.Constant):
            if new_test.value:
                ret = self.visit(node.body)
                self.switch_off_use()
                return ret
            else:
                ret = self.visit(node.orelse)
                self.switch_off_use()
                return ret
        new_if = ast.IfExp(test=new_test,
                           body=self.visit(node.body),
                           orelse=self.visit(node.orelse))
        self.switch_off_use()
        ast.copy_location(new_if, node)
        return new_if

    def visit_Assign(self, node):
        new_targets = self._process_list(node.targets)
        self.switch_on_use()
        new_value = self.visit(node.value)
        self.switch_off_use()
        reserved_targets = []
        reserved_values = []
        if isinstance(new_value, ast.Tuple):
            assert isinstance(new_targets[0], ast.Tuple), type(new_targets[0])
            for t, v in zip(new_targets[0].elts, new_value.elts):
                if not (isinstance(v, ast.Name) and v.id == t.id):
                    # avoid self-assignment
                    reserved_targets.append(t)
                    reserved_values.append(v)
                elif isinstance(v, ast.Constant):
                    self.defines[t.id] = v.value
            if len(reserved_targets) == 0:
                return None
            reserved_targets = [
                ast.Tuple(elts=reserved_targets, ctx=new_targets[0].ctx)
            ]
            reserved_values = ast.Tuple(elts=reserved_values,
                                        ctx=new_value.ctx)
        else:
            if isinstance(new_value,
                          ast.Name) and new_value.id == node.targets[0].id:
                # avoid self-assignment
                return None
            else:
                reserved_targets = new_targets
                reserved_values = new_value
                if isinstance(reserved_values, ast.Constant):
                    self.defines[
                        reserved_targets[0].id] = reserved_values.value
        new_assign = ast.Assign(targets=reserved_targets,
                                value=reserved_values)
        ast.copy_location(new_assign, node)
        return new_assign

    def visit_Attribute(self, node):

        def _recursive_visit_attribute(node):
            if isinstance(node, ast.Attribute):
                val = _recursive_visit_attribute(node.value)
                if isinstance(val, ast.AST):
                    return node
                # otherwise, try to get the attribute
                attr = getattr(val, node.attr)
                if hasattr(attr,
                           "__builtin__") and attr.__builtin__:  # builtin
                    # builtin function/class is reserved
                    return node
                else:
                    return attr
            if isinstance(node, ast.Name):
                if node.id in self.ctx:
                    return self.ctx[node.id]
            else:
                return node

        attr = _recursive_visit_attribute(node)
        if isinstance(attr, ast.AST):
            return node
        ret = ast.Constant(value=attr)
        ast.copy_location(ret, node)
        return ret

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if self.is_rvalue:
            if node.id in self.ctx:
                value = self.ctx[node.id]
                if isinstance(value, (int, float, str, LLType)):
                    return ast.Constant(value=value)
            if node.id in self.defines:
                value = self.defines[node.id]
                if isinstance(value, (int, float, str, LLType)):
                    return ast.Constant(value=value)
        return node

    def visit_FunctionDef(self, node):
        new_name = node.name  # Preserve unless mutated
        self.switch_on_use()
        new_args = self.visit(node.args)
        self.switch_off_use()
        new_body = self._process_list(node.body)
        new_decorator_list = self._process_list(node.decorator_list)
        self.switch_on_use()
        new_returns = self.visit(node.returns)
        self.switch_off_use()
        new_type_comment = node.type_comment  # Primitive, no mutation
        new_node = ast.FunctionDef(name=new_name,
                                   args=new_args,
                                   body=new_body,
                                   decorator_list=new_decorator_list,
                                   returns=new_returns,
                                   type_comment=new_type_comment)
        ast.copy_location(new_node, node)
        return new_node

    def visit_Call(self, node):
        self.switch_on_use()
        # new_func = self.visit(node.func)
        new_args = self._process_list(node.args)
        new_kwargs = self._process_list(node.keywords)
        self.switch_off_use()

        value = recursive_resolve_attribute(node.func, self.ctx)
        if isinstance(value, Callable):
            if hasattr(value, "__const_func__") and value.__const_func__:
                for arg in new_args:
                    assert isinstance(
                        arg, ast.Constant
                    ), f"Const function should only take Constant as arguments, but get {arg} {ast.unparse(arg)}"
                call_args = [arg.value for arg in new_args]
                kw_call_args = {}
                for kw in new_kwargs:
                    assert isinstance(kw, ast.keyword)
                    assert isinstance(
                        kw.value, ast.Constant
                    ), f"Const function should only take Constant as kw arguments, but get {kw.value}"
                    kw_call_args[kw.arg] = kw.value.value
                ret = value(*call_args, **kw_call_args)
                return ast.Constant(value=ret)
            elif hasattr(value,
                         "__const_type_func__") and value.__const_type_func__:
                assert node in self.all_types, f"Call node {ast.dump(node)} is not in all_types"
                ret_type = self.all_types[node]
                return ast.Constant(value=ret_type)
        new_call = ast.Call(func=node.func, args=new_args, keywords=new_kwargs)
        ast.copy_location(new_call, node)
        return new_call

    def visit_Assert(self, node):
        new_test = self.visit(node.test)
        new_msg = self.visit(node.msg) if node.msg else None

        assert isinstance(
            new_test, ast.Constant
        ), f"assert in kernel should be static, but the test results is {ast.dump(new_test)}"
        if not new_test.value:
            raise AssertionError(
                f"Compile-time Assert Failed:"
                f"\n{eval(ast.unparse(new_msg), self.ctx) if new_msg is not None else ''}\n"
                f"row:{node.lineno},col:{node.col_offset}:   {ast.unparse(node)}"
            )

        return None

    def visit_arg(self, node: ast.arg) -> ast.AST:
        node_copy = self._copy_node(node)
        node_copy.arg = node.arg  # String (parameter name)
        # keep annotation
        node_copy.annotation = self.visit(
            node.annotation) if node.annotation else None
        node_copy.type_comment = self.visit(node.type_comment)
        return node_copy

    def visit_Subscript(self, node):
        new_value = self.visit(node.value)
        new_slice = self.visit(node.slice)
        if isinstance(new_value, ast.Constant) and isinstance(
                new_slice, ast.Constant):
            return ast.Constant(value=new_value.value[new_slice.value])
        return ast.Subscript(value=new_value, slice=new_slice, ctx=node.ctx)


def const_fold(tree: ast.AST, ctx: Dict[str, Any] = None) -> ast.AST:
    if ctx is None:
        ctx = {}
    inferencer = TypeInferencer(ctx=ctx, scope_vars={})
    inferencer.visit(tree)
    all_types = inferencer.all_types
    return ast.fix_missing_locations(
        ConstantFolder(ctx=ctx, all_types=all_types).visit(tree))
