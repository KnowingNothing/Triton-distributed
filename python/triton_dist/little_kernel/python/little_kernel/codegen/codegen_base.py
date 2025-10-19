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
from io import StringIO
from typing import Dict, List, Optional, Union, Callable
import sys
import little_kernel.language as ll
from little_kernel.language.builtin_base import Builtin
from little_kernel.core.passes.utils.infer_type import TypeInferencer
from little_kernel.core.passes.utils.resolve_attribute import recursive_resolve_attribute
from little_kernel.core.compile import LLKernel
from little_kernel.core.passes import PASSES


class CppEmitter(ast.NodeVisitor):
    """
    C++ code emitter with context-aware LLType resolution and scope tracking.
    - Avoids redundant type declarations for in-scope variables (parameters/previous assignments)
    - Enforces basic type safety for reassignments
    - Resolves namespace aliases via user-provided context (e.g., 'lk.uint32')
    """

    def __init__(self, ctx: Dict[str, object], all_types=None):
        """
        Initialize the C++ emitter with a user-provided context.
        
        Args:
            ctx: A dictionary representing the namespace context (e.g., function.__globals__).
                 Must contain the alias used to import `little_kernel.language` (e.g., 'll' or 'lk').
        """
        # Code buffers (headers → structs → main code)
        self.header_buffer = StringIO()
        self.header_cache = set()
        self.struct_buffer = StringIO()
        self.main_buffer = StringIO()
        self.builtin_buffer = StringIO()
        self.builtin_cache = set()

        # State management
        self.indent_level = 0
        self.indent_unit = "    "  # 4-space indentation

        # LLType tracking: StructType → C++ struct name
        self.struct_types: Dict["ll.StructType", str] = {}

        # Context & annotation cache
        self.ctx = ctx  # User's namespace (for alias resolution)
        self.annotation_cache: Dict[ast.AST, "ll.LLType"] = {}

        # Scope tracking: Maps variable names to their LLType (per function scope)
        # Reset when entering/exiting a FunctionDef
        self.scope_vars: Dict[str, "ll.LLType"] = {}
        self.all_types = all_types if all_types is not None else {}

        # Add required C++ headers
        self._add_basic_headers()

        # Validate context (ensure LLType-related objects are present)
        self._validate_context()

    def _add_basic_headers(self) -> None:
        """Add C++ headers for std types."""
        headers = [
            "<cstdint>",  # For stdint types (uint32_t, int64_t)
            "<string>",  # For std::string (maps to StringType)
        ]
        for hdr in headers:
            self.header_cache.add(hdr)
            self.header_buffer.write(f"#include {hdr}\n")

    def _validate_context(self) -> None:
        """Validate context contains `little_kernel.language` or LLType objects."""
        has_lltype = any(
            # Check if object is the LLType module
            (hasattr(obj, "__name__")
             and obj.__name__.startswith("little_kernel.language"))
            # Check if object is an LLType instance
            or (isinstance(obj, (type, object)) and "LLType" in str(type(obj)))
            for obj in self.ctx.values())

        if not has_lltype:
            raise ValueError(
                "Context (ctx) missing `little_kernel.language` or LLType objects. "
                "Ensure ctx is the __globals__ of a function that imports `little_kernel.language`."
            )

    def indent_str(self) -> str:
        """Return current indentation string."""
        return self.indent_unit * self.indent_level

    # ------------------------------ Code Writing Helpers ------------------------------
    def write_main(self, s: str) -> None:
        """Write to main code buffer (functions, classes, statements)."""
        self.main_buffer.write(s)

    def writeln_main(self, s: str = "") -> None:
        """Write line to main code buffer with newline."""
        self.write_main(f"{self.indent_str()}{s}\n")

    def writeln_header(self, s: str = "") -> None:
        """Write line to header buffer with newline."""
        self.header_buffer.write(f"{s}\n")

    def write_struct(self, s: str) -> None:
        """Write to struct definition buffer."""
        self.struct_buffer.write(s)

    def writeln_struct(self, s: str = "") -> None:
        """Write line to struct buffer with newline."""
        self.write_struct(f"{s}\n")

    def writeln_builtin(self, s: str = "") -> None:
        """Write lines to builtin buffer with newline."""
        self.builtin_buffer.write(f"{s}\n")

    # ------------------------------ LLType to C++ Mapping ------------------------------
    def _generate_struct_name(self, struct_type: "ll.StructType") -> str:
        """Generate a unique, valid C++ struct name from StructType fields."""
        field_suffix = "_".join([
            f"{name}_{self._lltype_to_cpp(typ)}"
            for name, typ in struct_type.type_tuple
        ])
        # Sanitize for C++ identifier rules (replace invalid chars)
        valid_suffix = (field_suffix.replace("<", "_").replace(
            ">", "_").replace("*", "_ptr_").replace(" ", ""))
        return f"Struct_{valid_suffix}"

    def _lltype_to_cpp(self, ll_type: "ll.LLType") -> str:
        """Convert an LLType instance to a valid C++ type string."""
        # ------------------------------ Scalar Types ------------------------------
        if hasattr(ll_type, "kind") and ll_type.kind == "int":  # IntType
            if ll_type.special == "bool":
                return "bool"
            if ll_type.special == "binary":
                return "uint4"  # CUDA-compatible 4-bit binary type
            # Map to stdint types (e.g., uint32 → uint32_t)
            sign_prefix = "u" if not ll_type.signed else ""
            bit_map = {
                4: f"{sign_prefix}int4",
                8: f"{sign_prefix}int8_t",
                16: f"{sign_prefix}int16_t",
                32: f"{sign_prefix}int32_t",
                64: f"{sign_prefix}int64_t",
                128: f"unsigned __int128" if not ll_type.signed else "__int128"
            }
            return bit_map[ll_type.bits]

        elif hasattr(ll_type, "kind") and ll_type.kind == "float":  # FloatType
            float_map = {
                "fp4_e2m1": "__fp4",
                "fp8_e5m2": "__fp8_e5m2",
                "fp8_e4m3": "__fp8_e4m3",
                "bfloat16": "__bf16",
                "float16": "__half",
                "tfloat32": "__tfloat32",
                "float32": "float",
                "float64": "double"
            }
            return float_map[ll_type.fmt]

        elif hasattr(ll_type, "kind") and ll_type.kind == "void":  # VoidType
            return "void"

        elif hasattr(ll_type, "kind") and ll_type.kind == "str":  # StringType
            return "std::string"

        # ------------------------------ Annotated Types ------------------------------
        elif ll_type.is_const():  # Const/GridConstant
            inner_cpp = self._lltype_to_cpp(ll_type.inner_type)
            return f"__grid_constant__ {inner_cpp}" if ll_type.is_grid_constant(
            ) else f"const {inner_cpp}"

        elif ll_type.is_pointer():  # Pointer
            inner_cpp = self._lltype_to_cpp(ll_type.inner_type)
            return f"{inner_cpp}*"

        # ------------------------------ Composite Types ------------------------------
        elif ll_type.is_tensor():  # TensorType (map to pointer for simplicity)
            elem_cpp = self._lltype_to_cpp(ll_type.element_type)
            return f"{elem_cpp}*"

        elif ll_type.is_struct():  # StructType
            if ll_type not in self.struct_types:
                self.struct_types[ll_type] = self._generate_struct_name(
                    ll_type)
            return self.struct_types[ll_type]

        elif str(ll_type) == "TmaDescriptor":  # TmaDescriptorType
            return "cute::TmaDescriptor"

        elif str(ll_type) == "Wgmma":  # WgmmaType
            raise NotImplementedError(
                "WgmmaType is not supported for C++ generation")

        # ------------------------------ Unsupported LLType ------------------------------
        else:
            raise NotImplementedError(
                f"Unsupported LLType: {type(ll_type).__name__} (value: {str(ll_type)})"
            )

    # ------------------------------ Struct Generation ------------------------------
    def _emit_struct_definitions(self) -> None:
        """Auto-generate C++ structs for all tracked StructType instances."""
        if not self.struct_types:
            return

        self.writeln_struct("// Auto-generated structs from LLType StructType")
        for ll_struct, cpp_name in self.struct_types.items():
            self.writeln_struct(f"struct {cpp_name} {{")
            # Emit struct fields (name + C++ type)
            for field_name, field_type in ll_struct.type_tuple:
                field_cpp = self._lltype_to_cpp(field_type)
                self.writeln_struct(f"    {field_cpp} {field_name};")
            self.writeln_struct("};")
            self.writeln_struct()  # Blank line between structs

    # ------------------------------ AST Node Visitors ------------------------------
    def get_type(self, value):
        """Get the LLType of a value."""
        if isinstance(value, ll.LLType):
            return value
        elif isinstance(value, ast.Constant) and isinstance(
                value.value, ll.LLType):
            return value.value
        assert value in self.all_types, f"Value {ast.dump(value) if isinstance(value, ast.AST) else value} is not in all_types"
        return self.all_types[value]

    def visit_Module(self, node: ast.Module) -> None:
        """Process top-level module (traverse all statements)."""
        for stmt in node.body:
            self.visit(stmt)
            self.writeln_main()  # Blank line between top-level elements

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Process function definition: initialize scope with parameters, then process body."""
        # Resolve return type
        return_lltype = self.get_type(node.returns)
        return_cpp = self._lltype_to_cpp(return_lltype)

        # Resolve parameters and populate initial scope
        cpp_params = []
        old_scope_vars = self.scope_vars
        self.scope_vars = {}  # Reset scope for new function
        for arg in node.args.args:
            arg_lltype = self.get_type(arg.annotation)
            arg_cpp = self._lltype_to_cpp(arg_lltype)
            cpp_params.append(f"{arg_cpp} {arg.arg}")
            # Add parameter to scope (prevents re-declaration in body)
            self.scope_vars[arg.arg] = arg_lltype

        # Write function signature
        self.writeln_main(
            f"{return_cpp} {node.name}({', '.join(cpp_params)}) {{")
        self.indent_level += 1

        # Process function body (statements like Assign, Return)
        for stmt in node.body:
            self.visit(stmt)

        # Cleanup: close function and reset scope
        self.indent_level -= 1
        self.writeln_main("}")
        self.writeln_main("")
        self.scope_vars = old_scope_vars  # Restore scope after function

    def visit_Assign(self, node: ast.Assign) -> None:
        """
        Process assignment (single-target or tuple unpacking).
        - Skips type declaration for in-scope variables.
        - Enforces type matching for reassignments.
        """
        if len(node.targets) != 1:
            raise NotImplementedError(
                "Only single-target assignments are supported")

        target = node.targets[0]
        value_node = node.value

        # ------------------------------ Single Name Target (e.g., "x = 5" or "x: lk.int32 = 5") ------------------------------
        if isinstance(target, ast.Name):
            var_name = target.id
            value_cpp = self.visit(value_node)
            value_to_infer = value_cpp if isinstance(value_cpp,
                                                     ast.AST) else value_node

            # Case 1: Variable is already in scope (parameter or previous assignment)
            if var_name in self.scope_vars:
                # Type check: ensure new value matches existing type
                existing_type = self.scope_vars[var_name]
                new_value_type = self.get_type(value_to_infer)
                if existing_type != new_value_type:
                    self.writeln_main(
                        f"{var_name} = static_cast<{self._lltype_to_cpp(existing_type)}>({value_cpp});"
                    )
                else:
                    # Generate type-free assignment
                    self.writeln_main(f"{var_name} = {value_cpp};")

            # Case 2: Variable is new (not in scope)
            else:
                var_lltype = self.get_type(value_to_infer)
                var_cpp_type = self._lltype_to_cpp(var_lltype)
                # Generate declaration + assignment
                self.writeln_main(f"{var_cpp_type} {var_name} = {value_cpp};")
                # Add new variable to scope
                self.scope_vars[var_name] = var_lltype

        # ------------------------------ Tuple Unpacking Target (e.g., "(x, y) = (5, 3.14)") ------------------------------
        elif isinstance(target, ast.Tuple):
            # Validate tuple length matches value length
            if not isinstance(value_node, ast.Tuple) or len(
                    target.elts) != len(value_node.elts):
                raise ValueError(
                    f"Tuple unpacking mismatch: target has {len(target.elts)} elements, "
                    f"value has {len(value_node.elts) if isinstance(value_node, ast.Tuple) else 1} elements"
                )

            # Process each element in the tuple
            for target_elt, value_elt in zip(target.elts, value_node.elts):
                if not isinstance(target_elt, ast.Name):
                    raise NotImplementedError(
                        f"Only Name elements are supported in tuple unpacking (got {type(target_elt).__name__})"
                    )
                elt_name = target_elt.id
                elt_cpp = self.visit(value_elt)
                elt_to_infer = elt_cpp if isinstance(elt_cpp,
                                                     ast.AST) else value_elt

                # Case 1: Element is in scope
                if elt_name in self.scope_vars:
                    existing_type = self.scope_vars[elt_name]
                    new_elt_type = self.get_type(elt_to_infer)
                    if existing_type != new_elt_type:
                        self.writeln_main(
                            f"{elt_name} = static_cast<{existing_type}>({elt_cpp});"
                        )
                    else:
                        self.writeln_main(f"{elt_name} = {elt_cpp};")

                # Case 2: Element is new
                else:
                    elt_lltype = self.get_type(elt_to_infer)
                    elt_cpp_type = self._lltype_to_cpp(elt_lltype)
                    self.writeln_main(
                        f"{elt_cpp_type} {elt_name} = {elt_cpp};")
                    self.scope_vars[elt_name] = elt_lltype

        # ------------------------------ Unsupported Target Type ------------------------------
        else:
            raise NotImplementedError(
                f"Unsupported assignment target type: {type(target).__name__}\n"
                f"Node details: {ast.dump(target, indent=2)}")

    def visit_Name(self, node: ast.Name) -> str:
        """Process variable/identifier reference (return as-is for C++)."""
        if node.id in self.scope_vars:
            return node.id
        if node.id in self.ctx:
            return ast.Constant(value=self.ctx[node.id])
        return node.id

    def visit_Constant(self, node: ast.Constant) -> str:
        """Process constants (map to C++ syntax)."""
        value = node.value
        if isinstance(value, str):
            if value.startswith("0x"):
                return value  # C++ hex literals (e.g., 0x1234)
            return f'"{value}"'  # C++ string literals (double quotes)
        elif isinstance(value, bool):
            return "true" if value else "false"  # C++ lowercase bools
        elif isinstance(value, (int, float)):
            return str(value)
        elif isinstance(value, ll.LLType):
            return self._lltype_to_cpp(value)
        else:
            raise NotImplementedError(
                f"Unsupported constant type: {type(value).__name__} (value: {value})"
            )

    def visit_Expr(self, node: ast.Expr) -> None:
        """Process expression statement (add C++ semicolon)."""
        expr_cpp = self.visit(node.value)
        self.writeln_main(f"{expr_cpp};")

    def visit_Attribute(self, node: ast.Attribute):

        attr = recursive_resolve_attribute(node, self.ctx)
        if isinstance(attr, ast.AST):
            return node
        ret = ast.Constant(value=attr)
        ast.copy_location(ret, node)
        return ret

    def visit_Call(self, node: ast.Call) -> str:
        """Process function calls (special case: Python print → C++ std::cout)."""
        func = self.visit(node.func)

        if isinstance(func, ast.Constant):
            func_name = func.value
            # Map Python print() to C++ std::cout
            if func_name == "print":
                args_cpp = [self.visit(arg) for arg in node.args]
                return f"printf({', '.join(args_cpp)})"
            if isinstance(func_name, Callable):
                if hasattr(func_name, "__builtin__"):
                    assert hasattr(func_name, "__codegen_func__")
                    args_cpp = [self.visit(arg) for arg in node.args]
                    kwargs_cpp = {
                        kv.arg: self.visit(kv.value)
                        for kv in node.keywords
                    }
                    builtin = func_name.__codegen_func__(
                        *args_cpp, **kwargs_cpp)
                    assert isinstance(
                        builtin, Builtin
                    ), f"__codegen_func__ should return a Builtin object, but got {type(builtin).__name__}: {builtin}"
                    if func_name not in self.builtin_cache:
                        if builtin.body:
                            self.writeln_builtin(builtin.body)
                        self.builtin_cache.add(func_name)
                        for hd in builtin.includes:
                            if hd not in self.header_cache:
                                self.writeln_header(f"#include {hd}")
                                self.header_cache.add(hd)
                    return builtin.return_val
                elif isinstance(func_name, LLKernel):
                    # another kernel to compile
                    passes = PASSES[func_name.backend]
                    func_tree = func_name.lower(passes)
                    inferencer = TypeInferencer(func_name.ctx, {})
                    inferencer.visit(func_tree)
                    emitter = self.__class__(func_name.ctx,
                                             inferencer.all_types)
                    emitter.visit(func_tree)
                    func_cpp = emitter.get_code(need_header=False)
                    for hd in emitter.header_cache:
                        if hd not in self.header_cache:
                            self.writeln_header(f"#include {hd}")
                            self.header_cache.add(hd)
                    self.writeln_builtin(func_cpp)
                    args_cpp = [self.visit(arg) for arg in node.args]
                    assert len(
                        node.keywords
                    ) == 0, "Keyword arguments are not supported for recursive LLKernel call"
                    return f"{func_name.__name__}({', '.join(args_cpp)})"

        else:
            raise NotImplementedError(f"Unsupported function call: {func}")

    def visit_Return(self, node: ast.Return) -> None:
        """Process return statement."""
        if node.value is None:
            self.writeln_main("return;")
        else:
            value_cpp = self.visit(node.value)
            self.writeln_main(f"return {value_cpp};")

    def visit_BinOp(self, node: ast.BinOp) -> str:
        """Process binary operations (map to C++ operators)."""
        left_cpp = self.visit(node.left)
        op_cpp = self._get_operator_str(node.op)
        right_cpp = self.visit(node.right)
        return f"({left_cpp} {op_cpp} {right_cpp})"

    def _get_operator_str(self, op: ast.operator) -> str:
        """Map Python AST operators to C++ operators."""
        op_map = {
            ast.Add: "+",
            ast.Sub: "-",
            ast.Mult: "*",
            ast.Div: "/",
            ast.FloorDiv:
            "/",  # C++ uses / for integer division (adjust if needed)
            ast.Mod: "%",
            ast.Eq: "==",
            ast.NotEq: "!=",
            ast.Lt: "<",
            ast.LtE: "<=",
            ast.Gt: ">",
            ast.GtE: ">="
        }
        op_type = type(op)
        if op_type in op_map:
            return op_map[op_type]
        raise NotImplementedError(f"Unsupported operator: {op_type.__name__}")

    def generic_visit(self, node: ast.AST) -> Union[None, str]:
        """Default handler for unimplemented AST nodes."""
        # inline may result lists of stmts
        if isinstance(node, list):
            for stmt in node:
                self.visit(stmt)
        else:
            raise NotImplementedError(
                f"Unsupported AST node type: {type(node).__name__}\n"
                f"Node details: {ast.dump(node, indent=2)}")

    def visit_If(self, node: ast.If) -> None:
        """
        Process if-else statements.
        Generates C++ syntax:
        if (condition) {
            // body statements
        } else {
            // orelse statements (optional)
        }
        """
        # 1. Generate condition expression (e.g., "x > 5" → "(x > 5)")
        condition_cpp = self.visit(node.test)
        self.writeln_main(f"if ({condition_cpp}) {{")

        # 2. Process statements in if block (increase indentation)
        self.indent_level += 1
        for stmt in node.body:
            self.visit(stmt)
        self.indent_level -= 1
        self.writeln_main("}")  # Close if block

        # 3. Process else block (if exists)
        if node.orelse:
            # Check if else block contains a single if statement (i.e., "else if" case)
            if len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If):
                # Generate "else if (...)" syntax (no line break, direct concatenation)
                self.write_main(" else ")
                self.visit(node.orelse[0])  # Recursively process nested if
            else:
                # Generate regular else block
                self.writeln_main("else {")
                self.indent_level += 1
                for stmt in node.orelse:
                    self.visit(stmt)
                self.indent_level -= 1
                self.writeln_main("}")  # Close else block

    def visit_BoolOp(self, node: ast.BoolOp) -> str:
        """
        Process boolean operations (Python 'and'/'or' → C++ '&&'/'||').
        
        Converts expressions like `a and b or c` to `((a && b) || c)`.
        Wraps sub-expressions in parentheses to preserve operator precedence.
        """
        # Map Python boolean operators to C++ equivalents
        op_map = {
            ast.And: "&&",  # Python 'and' → C++ '&&'
            ast.Or: "||"  # Python 'or' → C++ '||'
        }
        op_type = type(node.op)
        if op_type not in op_map:
            raise NotImplementedError(
                f"Unsupported boolean operator: {op_type.__name__}")

        # Process each sub-expression in the boolean operation
        sub_exprs = [self.visit(expr) for expr in node.values]

        # Join sub-expressions with the C++ operator, wrap in parentheses
        # Example: [a, b, c] with '&&' → "(a && b && c)"
        return f"({f' {op_map[op_type]} '.join(sub_exprs)})"

    def visit_Compare(self, node: ast.Compare) -> str:
        """
        Process comparison expressions (e.g., `a < b <= c`, `x == 5 or y != 3`).
        
        Converts Python comparison chains (e.g., `a < b < c`) to C++-compatible 
        expressions with logical AND (`(a < b) && (b < c)`), preserving operator precedence.
        """
        # Process leftmost expression (e.g., "a" in "a < b < c")
        left_expr = self.visit(node.left)
        sub_conditions = []

        # Iterate over comparison operators and right-hand side expressions
        # For "a < b <= c", ops = [<, <=], comparators = [b, c]
        for op, comparator in zip(node.ops, node.comparators):
            # Get C++ operator string (e.g., Lt → "<")
            op_str = self._get_operator_str(op)
            # Process right-hand side expression (e.g., "b" or "c")
            right_expr = self.visit(comparator)
            # Wrap sub-condition in parentheses to preserve precedence
            sub_conditions.append(f"({left_expr} {op_str} {right_expr})")
            # Update left_expr to current comparator for chained comparisons
            left_expr = right_expr

        # Join chained conditions with "&&" (Python's a < b < c is equivalent to a < b and b < c)
        if len(sub_conditions) == 1:
            return sub_conditions[0]  # Single condition (e.g., "a == b")
        else:
            return f"({ ' && '.join(sub_conditions) })"  # Chained conditions

    def visit_For(self, node: ast.For) -> None:
        """
        Process Python for loops (primarily supports `for var in range(...)` → C++ for loop).
        Converts Python syntax:
        for target in range(start, stop, step):
            body
        To C++ syntax:
        for (type target = start; target < stop; target += step) {
            body
        }
        
        Notes:
        - Only supports `range()` as iterator (common in kernel code)
        - Supports single-variable targets (e.g., 'i' → not tuple unpacking yet)
        - Handles scope for loop variables (declares in loop init if new)
        """
        need_unroll = False
        # Detect ll.unroll wrapper (e.g., ll.unroll(range(4)))
        if isinstance(node.iter, ast.Call):
            func = recursive_resolve_attribute(node.iter.func, self.ctx)
            if isinstance(func,
                          Callable) and func.__name__ == ll.unroll.__name__:

                # Extract the inner range iterator
                unroll_arg = node.iter.args[0]
                need_unroll = True
                # Override iterator to the inner range for normal processing
                node.iter = unroll_arg

                # Evaluate range to get unroll count (must be constant)
                range_args = [self.visit(arg) for arg in unroll_arg.args]

        # 1. Validate loop target (only support single Name for now; tuple unpacking TBD)
        if not isinstance(node.target, ast.Name):
            raise NotImplementedError(
                f"For loop target must be a single variable (ast.Name), got {type(node.target).__name__}. "
                "Tuple unpacking (e.g., for (a,b) in ...) is not supported yet."
            )
        loop_var_name = node.target.id

        # 2. Validate & process iterator (only support `range()` call)
        if not (isinstance(node.iter, ast.Call) and isinstance(
                node.iter.func, ast.Name) and node.iter.func.id == "range"):
            raise NotImplementedError(
                f"For loop iterator must be `range()`, got {ast.dump(node.iter, indent=2)}. "
                "Other iterables (e.g., lists, tuples) are not supported yet.")
        range_call = node.iter

        # 3. Parse range arguments (start, stop, step) with defaults
        # Python range signature: range(stop) → start=0, step=1; range(start, stop, step=1)
        range_args = [self.visit(arg) for arg in range_call.args]
        # Get start, stop, step with defaults
        stop = range_args[0] if range_args else None
        start = range_args[1] if len(range_args) >= 2 else 0
        step = range_args[2] if len(range_args) >= 3 else 1

        # Validate step (prevent infinite loops)
        if step == 0:
            raise ValueError(f"Range step cannot be 0 (infinite loop)")

        # 4. Determine loop variable type & scope
        # Case A: Loop variable already exists in outer scope → reuse type, no redeclaration
        loop_var_in_scope = False
        if loop_var_name in self.scope_vars:
            loop_var_in_scope = True
            loop_var_type = self.scope_vars[loop_var_name]
            loop_var_cpp_type = self._lltype_to_cpp(loop_var_type)
            # In C++: use existing variable (no declaration in loop init)
            loop_init = f"{loop_var_name} = {start}"
        # Case B: New loop variable → declare in loop init (limit scope to loop)
        else:
            # Default to int32 for range loop variables (common in kernel code)
            loop_var_type = ll.int32
            loop_var_cpp_type = self._lltype_to_cpp(loop_var_type)
            # In C++: declare variable inside loop init
            loop_init = f"{loop_var_cpp_type} {loop_var_name} = {start}"
            # Temporarily add to scope_vars for body processing
            self.scope_vars[loop_var_name] = loop_var_type

        # 5. Generate loop condition (depends on step sign)
        if step > 0:
            loop_cond = f"{loop_var_name} < {stop}"
        else:  # step < 0 (counting down)
            loop_cond = f"{loop_var_name} > {stop}"

        # 6. Generate loop increment (C++: var += step)
        loop_incr = f"{loop_var_name} += {step}"

        # 7. Write C++ for loop header
        if need_unroll:
            self.writeln_main("#pragma unroll")
        self.writeln_main(f"for ({loop_init}; {loop_cond}; {loop_incr}) {{")

        # 8. Process loop body (increase indentation)
        self.indent_level += 1
        for stmt in node.body:
            self.visit(stmt)
        self.indent_level -= 1

        # 9. Close loop block
        self.writeln_main("}")

        # 10. Cleanup: Remove new loop variable from scope (if it was declared in loop)
        if not loop_var_in_scope:
            del self.scope_vars[loop_var_name]

        # 11. Handle orelse clause (C++ has no equivalent; raise error if non-empty)
        if node.orelse:
            raise NotImplementedError(
                "For loop 'orelse' clause is not supported in C++ (Python-specific feature). "
                "Remove the 'else' block from the for loop.")

    def visit_Pass(self, node: ast.Pass) -> None:
        """Process pass statement: no action needed."""
        pass
    
    def visit_Subscript(self, node: ast.Subscript) -> str:
        """
        Process subscript operations (array/tensor indexing, e.g., A[i], B[2][j], C[i][j][k]).
        Converts Python syntax to C++-compatible indexing:
        - Python: A[i] → C++: A[i]
        - Python: A[i][j] → C++: A[i][j]
        - Python: A[(i,j)] (multi-dim tuple index) → C++: A[i][j]
        - Python: A[5] (constant index) → C++: A[5]
        
        Notes:
        - Supports integer indices (constants, variables, nested subscripts)
        - Supports multi-dimensional indexing via tuples (e.g., A[i,j] → A[i][j])
        - Does NOT support slice indices (e.g., A[1:5], A[:]) yet (common in kernel code but TBD)
        """
        # 1. Process the base object being indexed (e.g., "A" in "A[i]")
        base_str = self.visit(node.value)
        if not base_str:
            raise RuntimeError(
                f"Invalid empty base object in subscript: {ast.dump(node.value, indent=2)}"
            )

        # 2. Process the index part (handle single/index tuple/multi-dim cases)
        index_node = node.slice
        index_strs: List[str] = []

        # Case A: Index is a tuple (multi-dimensional index, e.g., "(i,j)" in "A[i,j]")
        if isinstance(index_node, ast.Tuple):
            # Iterate over each element in the tuple (each dimension's index)
            for elem in index_node.elts:
                elem_str = self._process_single_index(elem)
                index_strs.append(elem_str)

        # Case B: Index is a single node (e.g., "i", "5", "B[k]")
        else:
            single_index_str = self._process_single_index(index_node)
            index_strs.append(single_index_str)

        # 3. Combine base + all indices into C++ indexing syntax (e.g., "A" + "[i]" + "[j]" → "A[i][j]")
        for idx_str in index_strs:
            base_str += f"[{idx_str}]"

        return base_str

    def _process_single_index(self, index_elem: ast.AST) -> str:
        """
        Helper to process a single index element (e.g., "i", "5", "B[k]").
        Validates that the index is an integer-compatible type (common in kernel indexing).
        """
        # Reject slice indices (e.g., A[1:5], A[:], A[::2]) – complex for kernel code, TBD
        if isinstance(index_elem, ast.Slice):
            raise NotImplementedError(
                f"Slice indices (e.g., A[1:5]) are not supported yet. "
                f"Found slice: {ast.dump(index_elem, indent=2)}")

        # Visit the index element to get its string representation
        index_str = self.visit(index_elem)

        # Validate the index type (should be integer-compatible, using existing type inference)
        index_type = self.get_type(index_elem)
        if not (isinstance(index_type, ll.IntType) and not index_type.special):
            raise TypeError(
                f"Subscript index must be a regular integer type (e.g., int32, uint64), "
                f"got {index_type} for index: {ast.dump(index_elem, indent=2)}"
            )

        return index_str

    # ------------------------------ Final Code Generation ------------------------------
    def get_code(self, need_header=True) -> str:
        """Combine headers, structs, and main code into final C++ output."""
        self._emit_struct_definitions()  # Emit structs before main code
        return (("/*start of generated code*/\n" +
                 self.header_buffer.getvalue() if need_header else "") + "\n" +
                self.struct_buffer.getvalue() + "\n" +
                self.builtin_buffer.getvalue() + "\n" +
                self.main_buffer.getvalue())


def codegen_cpp(tree: ast.AST, ctx=None, emit_header=True) -> str:
    if ctx is None:
        ctx = {}
    inferencer = TypeInferencer(ctx=ctx, scope_vars={})
    inferencer.visit(tree)
    all_types = inferencer.all_types
    emitter = CppEmitter(ctx=ctx, all_types=all_types)
    emitter.visit(tree)
    return emitter.get_code(need_header=emit_header)
