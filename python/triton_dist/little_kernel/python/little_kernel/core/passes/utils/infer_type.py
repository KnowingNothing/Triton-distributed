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
from typing import Dict, List, Optional, Callable, Any, Union
import little_kernel.language as ll
from little_kernel.language.builtin_base import Builtin
from little_kernel.core.type_system import *
from .resolve_attribute import recursive_resolve_attribute
from little_kernel.core.compile import LLKernel


class TypeInferenceError(RuntimeError):
    """Base exception for type inference failures (with AST node context)"""

    def __init__(self, message: str, node: Optional[ast.AST] = None):
        self.node_str = ast.dump(node,
                                 indent=2) if node else "No AST node provided"
        super().__init__(f"{message}\nAffected node:\n{self.node_str}")


class UndefinedVariableError(TypeInferenceError):
    """Raised when referencing a variable not in scope/context"""
    pass


class TypeMismatchError(TypeInferenceError):
    """Raised when types are incompatible (e.g., int + float)"""
    pass


class UnsupportedNodeError(TypeInferenceError):
    """Raised when an AST node type is not supported for inference"""
    pass


class TypeInferencer(ast.NodeVisitor):
    """
    Centralized type inferencer for LLType system.
    - Inherits from ast.NodeVisitor for structured AST traversal
    - Manages scope/variable types and annotation cache
    - Supports extended AST node types with isolated handlers
    """

    def __init__(self, ctx: Dict[str, Any], scope_vars):
        """
        Initialize type inferencer with global context.
        
        Args:
            ctx: Global namespace (e.g., function.__globals__) with LLType imports/constants
        """
        # State management
        self.ctx: Dict[
            str, Any] = ctx  # Global context (imports, top-level constants)
        self.scope_stack: List[Dict[str, LLType]] = [
            scope_vars
        ]  # Stack for nested scopes (function/loop)
        self.annotation_cache: Dict[ast.AST,
                                    LLType] = {}  # Cache evaluated annotations
        self.all_types = {}

        # Type handlers for operators (centralized for maintainability)
        self._bin_op_handlers: Dict[type,
                                    Callable[[LLType, LLType], LLType]] = {
                                        ast.Add: lambda a, b: a,
                                        ast.Sub: lambda a, b: a,
                                        ast.Mult: lambda a, b: a,
                                        ast.Div: lambda a, b: a,
                                        ast.FloorDiv: lambda a, b: a,
                                        ast.Mod: lambda a, b: a,
                                        ast.Pow: lambda a, b: a,
                                        ast.LShift: lambda a, b: a,
                                        ast.RShift: lambda a, b: a,
                                        ast.BitOr: lambda a, b: a,
                                        ast.BitXor: lambda a, b: a,
                                        ast.BitAnd: lambda a, b: a,
                                        ast.MatMult: lambda a, b: a,
                                    }

        self._un_op_handlers: Dict[type, Callable[[LLType], LLType]] = {
            ast.UAdd: lambda a: a,
            ast.USub: lambda a: a,
            ast.Invert: lambda a: a,
            ast.Not: lambda a: bool_,  # Logical NOT always returns bool
        }

        self._bool_op_handlers: Dict[type, Callable[[], LLType]] = {
            ast.And: lambda: bool_,
            ast.Or: lambda: bool_,
        }

    # ------------------------------ Scope Management Helpers ------------------------------
    @property
    def current_scope(self) -> Dict[str, LLType]:
        """Get the active scope (top of scope stack)"""
        return self.scope_stack[-1]

    def push_scope(self) -> None:
        """Create a new nested scope (e.g., for function bodies, loops)"""
        self.scope_stack.append({})

    def pop_scope(self) -> None:
        """Remove the active scope (e.g., when exiting a function/loop)"""
        if len(self.scope_stack) <= 1:
            raise RuntimeError("Cannot pop global scope")
        self.scope_stack.pop()

    def add_variable_to_scope(self, name: str, type_: LLType) -> None:
        """Add a variable to the active scope (fails if already exists)"""
        if name in self.current_scope:
            raise TypeInferenceError(
                f"Variable '{name}' already declared in current scope",
                node=None  # No specific node for duplicate declaration
            )
        self.current_scope[name] = type_

    def get_variable_type(self, name: str, node: ast.AST) -> LLType:
        """
        Get type of a variable from scope/context.
        Raises UndefinedVariableError if not found.
        """
        # 1. Check active scope first (function/loop variables)
        to_find_id = len(self.scope_stack) - 1
        scope_to_find = self.scope_stack[to_find_id]
        while to_find_id >= 0:
            if name in scope_to_find:
                return scope_to_find[name]
            to_find_id -= 1
            scope_to_find = self.scope_stack[to_find_id]

        # 2. Check global context (imports, top-level constants)
        if name in self.ctx:
            ctx_value = self.ctx[name]
            # Resolve LLType from context value
            if isinstance(ctx_value, LLType):
                return ctx_value
            elif isinstance(ctx_value, (int, float, str)):
                return self._infer_constant_type(ast.Constant(value=ctx_value))
            else:
                raise UndefinedVariableError(
                    f"Global context value '{name}' is not an LLType or primitive (got {type(ctx_value).__name__})",
                    node=node)

        # 3. Not found anywhere
        raise UndefinedVariableError(
            f"Variable '{name}' is not declared in scope or global context",
            node=node)

    # ------------------------------ Core Inference Helpers ------------------------------
    def _resolve_annotation(self, ann_node: ast.AST) -> LLType:
        """
        Evaluate an AST annotation node (e.g., `ll.Tensor[ll.int32]`) to LLType.
        Uses cache to avoid re-evaluation.
        """
        # Return cached result if available
        if ann_node in self.annotation_cache:
            return self.annotation_cache[ann_node]

        # Resolve annotation via attribute resolution (handles aliases like `lk.int32`)
        try:
            resolved = recursive_resolve_attribute(ann_node, self.ctx)
        except Exception as e:
            raise TypeInferenceError(
                f"Failed to resolve annotation (check aliases/imports): {str(e)}",
                node=ann_node) from e

        # Validate resolved value is LLType

        if isinstance(resolved, ast.Constant):
            resolved = resolved.value
        if not isinstance(resolved, LLType):
            raise TypeInferenceError(
                f"Annotation resolved to {type(resolved).__name__} (value: {resolved}), expected LLType",
                node=ann_node)

        # Cache and return
        self.annotation_cache[ann_node] = resolved
        return resolved

    def _infer_constant_type(self, node: ast.Constant) -> LLType:
        """Infer LLType for ast.Constant nodes (handles primitives and LLType constants)"""
        value = node.value
        if isinstance(value, int):
            return int32
        elif isinstance(value, float):
            return float32
        elif isinstance(value, str):
            return str_
        elif isinstance(value, LLType):
            return value
        else:
            raise UnsupportedNodeError(
                f"Unsupported constant type: {type(value).__name__} (value: {value})",
                node=node)

    def _validate_binary_op_types(self, left_type: LLType, right_type: LLType,
                                  node: ast.BinOp) -> None:
        """Ensure binary operands have matching types (C++ compatibility)"""
        if left_type != right_type:
            raise TypeMismatchError(
                f"Binary operation requires matching types: {left_type} (left) vs {right_type} (right)",
                node=node)

    # ------------------------------ AST Node Visitors ------------------------------
    def visit(self, node):
        if node in self.all_types:
            return self.all_types[node]
        if node is None:
            return None
        else:
            ret = super().visit(node)
            self.all_types[node] = ret
            return ret

    def visit_Module(self, node: ast.Module) -> None:
        """Process top-level module (traverse all statements in global scope)"""
        for stmt in node.body:
            self.visit(stmt)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """
        Process function definitions:
        1. Push new scope for function body
        2. Resolve parameters and add to scope
        3. Resolve return type and validate return statements
        4. Pop scope when done
        """
        # Resolve return type first
        if node.returns is None:
            raise TypeInferenceError(
                "Function requires return type annotation", node=node)
        return_type = self._resolve_annotation(node.returns)

        # Push new scope for function body
        self.push_scope()

        try:
            # Add parameters to function scope
            for arg in node.args.args:
                if arg.annotation is None:
                    raise TypeInferenceError(
                        f"Parameter '{arg.arg}' requires type annotation",
                        node=arg)
                arg_type = self._resolve_annotation(arg.annotation)
                self.add_variable_to_scope(arg.arg, arg_type)

            # Track if return statements match return type
            has_return = False
            for stmt in node.body:
                self.visit(stmt)
                # Validate return statements
                if isinstance(stmt, ast.Return):
                    has_return = True
                    if stmt.value is None:
                        if return_type != void:
                            raise TypeMismatchError(
                                f"Return statement has no value, but function return type is {return_type}",
                                node=stmt)
                    else:
                        return_value_type = self.infer_node_type(stmt.value)
                        if return_value_type != return_type:
                            raise TypeMismatchError(
                                f"Return value type {return_value_type} does not match function return type {return_type}",
                                node=stmt)

            # Add function itself to parent scope (for recursive calls)
            self.scope_stack[-2][
                node.
                name] = return_type  # Add to parent scope, not function scope

        finally:
            # Always pop scope, even if errors occur
            self.pop_scope()

    def visit_Assign(self, node: ast.Assign) -> None:
        """
        Process assignment statements:
        - Supports single variables, tuple unpacking, and subscript targets
        - Infers type from annotation (preferred) or value
        - Validates type consistency for reassignments
        """
        if len(node.targets) != 1:
            raise UnsupportedNodeError(
                "Only single-target assignments are supported (no multiple targets like 'a = b = 5')",
                node=node)

        target = node.targets[0]
        value_type = self.infer_node_type(node.value)

        # Case 1: Target is a single variable (e.g., "x = 5" or "x: ll.int32 = 5")
        if isinstance(target, ast.Name):
            var_name = target.id
            # Get target type from annotation (if exists)
            if hasattr(target, "annotation") and target.annotation is not None:
                target_type = self._resolve_annotation(target.annotation)
                # Validate annotation matches value type
                if target_type != value_type:
                    raise TypeMismatchError(
                        f"Assignment annotation type {target_type} does not match value type {value_type}",
                        node=node)
            else:
                # Infer type from value (fallback)
                target_type = value_type

            # Add to scope if new, or validate reassignement
            if var_name not in self.current_scope:
                self.add_variable_to_scope(var_name, target_type)
            else:
                existing_type = self.get_variable_type(var_name, node)
                if existing_type != target_type:
                    raise TypeMismatchError(
                        f"Reassignment to '{var_name}' changes type: {existing_type} → {target_type}",
                        node=node)

        # Case 2: Target is tuple unpacking (e.g., "(x, y) = (5, 3.14)")
        elif isinstance(target, ast.Tuple):
            # Validate value is a tuple with matching length
            if not (isinstance(node.value, ast.Tuple)
                    and len(target.elts) == len(node.value.elts)):
                raise TypeMismatchError(
                    f"Tuple unpacking length mismatch: target has {len(target.elts)} elements, value has {len(node.value.elts) if isinstance(node.value, ast.Tuple) else 1}",
                    node=node)
            # Process each element in the tuple
            for target_elt, value_elt in zip(target.elts, node.value.elts):
                if not isinstance(target_elt, ast.Name):
                    raise UnsupportedNodeError(
                        f"Tuple unpacking only supports variable targets (got {type(target_elt).__name__})",
                        node=target_elt)
                # Recursively infer element type and assign
                elt_value_type = self.infer_node_type(value_elt)
                elt_name = target_elt.id
                if elt_name not in self.current_scope:
                    self.add_variable_to_scope(elt_name, elt_value_type)
                else:
                    existing_type = self.get_variable_type(elt_name, node)
                    if existing_type != elt_value_type:
                        raise TypeMismatchError(
                            f"Tuple unpacking changes type of '{elt_name}': {existing_type} → {elt_value_type}",
                            node=node)

        # Case 3: Target is subscript (e.g., "A[i] = 5")
        elif isinstance(target, ast.Subscript):
            # Infer base type (e.g., A is Tensor[int32])
            base_type = self.infer_node_type(target.value)
            # Validate base is a tensor/array type (supports assignment)
            if not (base_type.is_tensor() or base_type.is_pointer()):
                raise TypeMismatchError(
                    f"Subscript assignment requires tensor/pointer type (got {base_type})",
                    node=target)
            # Validate value type matches tensor element type
            if base_type.is_tensor():
                expected_type = base_type.element_type
            else:  # Pointer type
                expected_type = base_type.inner_type
            if value_type != expected_type:
                raise TypeMismatchError(
                    f"Subscript assignment type {value_type} does not match base element type {expected_type}",
                    node=node)

        # Case 4: Unsupported target type
        else:
            raise UnsupportedNodeError(
                f"Unsupported assignment target type: {type(target).__name__}",
                node=target)

    def visit_For(self, node: ast.For) -> None:
        """
        Process for loops (supports `for var in range(...)` and `ll.unroll(range(...))`):
        1. Resolve loop variable type (infer or default to int32)
        2. Push temporary scope for loop body
        3. Add loop variable to scope
        4. Validate iterator is range-based
        5. Pop scope after loop
        """
        # Resolve iterator (handle ll.unroll wrapper)
        iter_node = node.iter
        if isinstance(iter_node, ast.Call):
            try:
                # Check if iterator is ll.unroll(...)
                resolved_func = recursive_resolve_attribute(
                    iter_node.func, self.ctx)
                if isinstance(
                        resolved_func, Callable
                ) and resolved_func.__name__ == ll.unroll.__name__:
                    if len(iter_node.args) != 1:
                        raise TypeInferenceError(
                            f"ll.unroll requires exactly 1 argument (got {len(iter_node.args)})",
                            node=iter_node)
                    iter_node = iter_node.args[0]  # Unwrap to inner range
            except Exception:
                pass  # Not a ll.unroll call, proceed with original iterator

        # Validate iterator is range()
        if not (isinstance(iter_node, ast.Call) and isinstance(
                iter_node.func, ast.Name) and iter_node.func.id == "range"):
            raise UnsupportedNodeError(
                f"For loop iterator must be range() or ll.unroll(range()) (got {type(iter_node).__name__})",
                node=iter_node)

        # Validate loop target (only single variable supported)
        if not isinstance(node.target, ast.Name):
            raise UnsupportedNodeError(
                f"For loop target must be a single variable (got {type(node.target).__name__})",
                node=node.target)
        loop_var_name = node.target.id

        # Infer loop variable type (from annotation or default to int32)
        if hasattr(node.target,
                   "annotation") and node.target.annotation is not None:
            loop_var_type = self._resolve_annotation(node.target.annotation)
            if not (isinstance(loop_var_type, ll.IntType)
                    and not loop_var_type.special):
                raise TypeMismatchError(
                    f"Loop variable must be a regular integer type (got {loop_var_type})",
                    node=node.target)
        else:
            loop_var_type = int32  # Default to int32 for range loops

        # Push scope for loop body (isolates loop variable)
        self.push_scope()
        try:
            # Add loop variable to loop scope
            self.add_variable_to_scope(loop_var_name, loop_var_type)
            # Process loop body
            for stmt in node.body:
                self.visit(stmt)
            # Process orelse (Python-specific, no type checks needed)
            for stmt in node.orelse:
                self.visit(stmt)
        finally:
            self.pop_scope()

    def visit_Subscript(self, node: ast.Subscript) -> LLType:
        """
        Infer type of subscript expression (e.g., "A[i]" → Tensor element type).
        Validates:
        - Base is tensor/pointer type
        - Indices are integer types
        """
        # Infer base type (e.g., A is Tensor[int32] or int32*)
        base_type = self.infer_node_type(node.value)
        if not (base_type.is_tensor() or base_type.is_pointer()
                or base_type.is_generic()):
            raise TypeMismatchError(
                f"Subscript requires tensor or pointer or generic base type (got {base_type})",
                node=node)

        # Validate indices are integers
        index_node = node.slice
        if isinstance(index_node, ast.Tuple):
            indices = index_node.elts
        else:
            indices = [index_node]
        eval_generic_type = base_type
        for idx in indices:
            idx_type = self.infer_node_type(idx)
            if not base_type.is_generic():
                if not (isinstance(idx_type, ll.IntType)
                        and not idx_type.special):
                    raise TypeMismatchError(
                        f"Subscript index must be regular integer (got {idx_type})",
                        node=idx)
            else:
                eval_generic_type = eval_generic_type[idx_type]

        # Return element type of the base
        if base_type.is_tensor():
            ret_type = base_type.element_type
        elif base_type.is_pointer():  # Pointer type
            ret_type = base_type.inner_type
        elif base_type.is_generic():  # Generic type
            ret_type = eval_generic_type
        else:
            raise UnsupportedNodeError(
                f"Unsupported subscript node: {ast.dump(node)}")
        self.all_types[node] = ret_type
        return ret_type

    def visit_Constant(self, node: ast.Constant) -> LLType:
        """Infer type of ast.Constant nodes"""
        const_type = self._infer_constant_type(node)
        self.all_types[node] = const_type
        return const_type

    def visit_Name(self, node: ast.Name) -> LLType:
        """Infer type of variable references (ast.Name)"""
        name_type = self.get_variable_type(node.id, node)
        self.all_types[node] = name_type
        return name_type

    def visit_BinOp(self, node: ast.BinOp) -> LLType:
        """Infer type of binary operations (e.g., x + y)"""
        left_type = self.infer_node_type(node.left)
        right_type = self.infer_node_type(node.right)

        # Validate operand types match
        self._validate_binary_op_types(left_type, right_type, node)

        # Get handler for operator type
        op_type = type(node.op)
        if op_type not in self._bin_op_handlers:
            raise UnsupportedNodeError(
                f"Unsupported binary operator: {op_type.__name__}", node=node)

        # Return result type from handler
        bin_type = self._bin_op_handlers[op_type](left_type, right_type)
        self.all_types[node] = bin_type
        return bin_type

    def visit_UnaryOp(self, node: ast.UnaryOp) -> LLType:
        """Infer type of unary operations (e.g., -x, not x)"""
        operand_type = self.infer_node_type(node.operand)

        # Get handler for operator type
        op_type = type(node.op)
        if op_type not in self._un_op_handlers:
            raise UnsupportedNodeError(
                f"Unsupported unary operator: {op_type.__name__}", node=node)

        # Return result type from handler
        un_type = self._un_op_handlers[op_type](operand_type)
        self.all_types[node] = un_type
        return un_type

    def visit_BoolOp(self, node: ast.BoolOp) -> LLType:
        """Infer type of boolean operations (and/or)  always returns bool"""
        # Validate all operands are boolean
        for value in node.values:
            value_type = self.infer_node_type(value)
            if not isinstance(value_type, (ScalarType, Pointer)):
                raise TypeMismatchError(
                    f"Boolean operation requires bool operand (got {value_type})",
                    node=value)

        # Get handler for operator type
        op_type = type(node.op)
        if op_type not in self._bool_op_handlers:
            raise UnsupportedNodeError(
                f"Unsupported boolean operator: {op_type.__name__}", node=node)

        # Boolean ops always return bool
        bool_type = self._bool_op_handlers[op_type]()
        self.all_types[node] = bool_type
        return bool_type

    def visit_Compare(self, node: ast.Compare) -> LLType:
        """Infer type of comparison operations (e.g., x < y)  always returns bool"""
        # Infer type of left-hand side
        left_type = self.infer_node_type(node.left)

        # Validate all comparators match left type
        for comparator in node.comparators:
            comp_type = self.infer_node_type(comparator)
            if comp_type != left_type:
                raise TypeMismatchError(
                    f"Comparison requires matching types: {left_type} (left) vs {comp_type} (comparator)",
                    node=node)

        # Comparisons always return bool
        self.all_types[node] = bool_
        return bool_

    def visit_Call(self, node: ast.Call) -> LLType:
        """Infer type of function calls (supports builtins, LLKernel, and __eval_return_type__ functions)"""
        # Resolve the called function
        try:
            func = recursive_resolve_attribute(node.func, self.ctx)
        except Exception as e:
            raise TypeInferenceError(
                f"Failed to resolve function for call: {str(e)}",
                node=node.func) from e

        # Case 1: Function has __eval_return_type__ (builtin with type hints)
        if isinstance(func, Callable) and hasattr(func,
                                                  "__eval_return_type__"):
            # Get return type from handler
            try:
                # Some intrins may put return values as arguments
                # First evaluate these return values
                if hasattr(func, "__eval_arg_type__"):
                    args = []
                    for arg in node.args:
                        if isinstance(arg, ast.Name):
                            args.append(arg.id)
                        elif isinstance(arg, ast.Constant):
                            args.append(arg.value)
                        else:
                            args.append(arg)
                    kwargs = {}
                    for kv in node.keywords:
                        if isinstance(kv.value, ast.Name):
                            kwargs[kv.arg] = kv.value.id
                        elif isinstance(kv.value, ast.Constant):
                            kwargs[kv.arg] = kv.value.value
                        else:
                            kwargs[kv.arg] = kv.value
                    func.__eval_arg_type__(self.current_scope, *args, **kwargs)
                # Infer argument types
                arg_types = [self.infer_node_type(arg) for arg in node.args]
                kwarg_types = {
                    kv.arg: self.infer_node_type(kv.value)
                    for kv in node.keywords
                }
                call_type = func.__eval_return_type__(*arg_types,
                                                      **kwarg_types)
                self.all_types[node] = call_type
                return call_type
            except Exception as e:
                raise TypeInferenceError(
                    f"Failed to compute return type for function {func.__name__}: {str(e)}",
                    node=node) from e

        # Case 2: Function is an LLKernel (nested kernel)
        elif isinstance(func, LLKernel):
            # Infer kernel return type via its own type inference
            kernel_scope = infer_type(func.lower([]), func.ctx)
            kernel_return_type = kernel_scope.get(func.py_func.__name__)
            if kernel_return_type is None:
                raise TypeInferenceError(
                    f"Could not infer return type for LLKernel {func.py_func.__name__}",
                    node=node)
            self.all_types[node] = kernel_return_type
            return kernel_return_type

        # Case 3: Unsupported function type
        else:
            raise UnsupportedNodeError(
                f"Unsupported function type for call: {type(func).__name__} (func: {func})",
                node=node)

    def visit_List(self, node: ast.List) -> LLType:
        """Infer type of lists (assumes homogeneous elements → Tensor[element_type])"""
        if not node.elts:
            raise TypeInferenceError(
                "Empty lists are not supported for type inference", node=node)
        # Infer element type from first element (enforce homogeneity)
        elem_type = self.infer_node_type(node.elts[0])
        for elem in node.elts[1:]:
            current_elem_type = self.infer_node_type(elem)
            if current_elem_type != elem_type:
                raise TypeMismatchError(
                    f"List has mixed types: {elem_type} (first) vs {current_elem_type} (later)",
                    node=node)
        # Store list type in scope
        self.all_types[node] = Tensor[elem_type]
        return Tensor[elem_type]

    def visit_Tuple(self, node: ast.Tuple) -> LLType:
        """Infer type of tuples (assumes homogeneous elements → Tensor[element_type])"""
        if not node.elts:
            raise TypeInferenceError(
                "Empty tuples are not supported for type inference", node=node)
        # Infer element type from first element (enforce homogeneity)
        elem_type = self.infer_node_type(node.elts[0])
        for elem in node.elts[1:]:
            current_elem_type = self.infer_node_type(elem)
            if current_elem_type != elem_type:
                raise TypeMismatchError(
                    f"Tuple has mixed types: {elem_type} (first) vs {current_elem_type} (later)",
                    node=node)
        # Store tuple type in scope
        self.all_types[node] = Tensor[elem_type]
        return Tensor[elem_type]

    def visit_ListComp(self, node: ast.ListComp) -> LLType:
        """Infer type of list comprehensions (→ Tensor[element_type])"""
        # Infer type from comprehension element
        elem_type = self.infer_node_type(node.elt)
        # Process generators (validate loop variables are integers)
        for gen in node.generators:
            if isinstance(gen, ast.comprehension):
                # Validate iterator is range-based
                if not (isinstance(gen.iter, ast.Call)
                        and isinstance(gen.iter.func, ast.Name)
                        and gen.iter.func.id == "range"):
                    raise UnsupportedNodeError(
                        f"List comprehension iterator must be range() (got {type(gen.iter).__name__})",
                        node=gen.iter)
                # Validate loop target is a variable
                if isinstance(gen.target, ast.Name):
                    # Add loop variable to temporary scope (for element type checks)
                    self.current_scope[
                        gen.target.id] = int32  # Range variables are int32
                else:
                    raise UnsupportedNodeError(
                        f"List comprehension target must be a single variable (got {type(gen.target).__name__})"
                    )
        # Store list comprehension type in scope
        self.all_types[node] = Tensor[elem_type]
        return Tensor[elem_type]

    def visit_Expr(self, node: ast.Expr) -> None:
        """Process expression statements (side effects, no type to return)"""
        self.infer_node_type(
            node.value)  # Infer type to validate, but no return

    def visit_Return(self, node: ast.Return) -> None:
        """Process return statements (type validation is done in visit_FunctionDef)"""
        if node.value is not None:
            self.infer_node_type(node.value)  # Validate value is inferable

    def visit_Attribute(self, node: ast.Attribute) -> LLType:
        tmp = recursive_resolve_attribute(node, self.ctx)
        if not isinstance(tmp, ast.AST):
            attr_type = self.infer_node_type(ast.Constant(value=tmp))
            self.all_types[node] = attr_type
            return attr_type
        else:
            raise RuntimeError(f"Can't evaluate Attributed {ast.dump(node)}")

    def visit_IfExp(self, node: ast.Attribute) -> LLType:
        body_type = self.visit(node.body)
        orelse_type = self.visit(node.orelse)
        if body_type != orelse_type:
            raise TypeMismatchError(
                f"IfExp has mixed types: {body_type} (body) vs {orelse_type} (orelse)",
                node=node)
        self.all_types[node] = body_type
        return body_type

    # ------------------------------ Public API ------------------------------
    def infer_node_type(self, node: ast.AST) -> LLType:
        """
        Public method to infer type of a single AST node.
        Delegates to the appropriate visit_* method.
        """
        if node in self.all_types:
            return self.all_types[node]
        result = self.visit(node)
        if not isinstance(result, LLType):
            raise TypeInferenceError(
                f"Node inference returned non-LLType: {type(result).__name__} (node: {ast.dump(node)})",
                node=node)
        self.all_types[node] = result
        return result

    def get_inferred_scopes(self) -> List[Dict[str, LLType]]:
        """Get all scopes with inferred variable types (for debugging/validation)"""
        return self.scope_stack.copy()


# ------------------------------ Top-Level API ------------------------------
def infer_type(tree: ast.AST,
               ctx: Dict[str, Any],
               scope_vars=None) -> Dict[str, LLType]:
    """
    Top-level function to run type inference on an AST.
    
    Args:
        tree: AST to infer types for
        ctx: Global context (e.g., function.__globals__)
    
    Returns:
        Global scope with inferred variable/function types
    """
    scope_vars = scope_vars if scope_vars is not None else {}
    inferencer = TypeInferencer(ctx, scope_vars)
    inferencer.visit(tree)
    # Return global scope (first element of scope stack)
    return inferencer.scope_stack[0]
