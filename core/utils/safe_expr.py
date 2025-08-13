# core/utils/safe_expr.py
from __future__ import annotations
import ast
from typing import Any, Dict

# Define allowed AST node types to prevent unsafe operations
ALLOWED_NODES = {
    ast.Expression, ast.BoolOp, ast.UnaryOp,
    ast.And, ast.Or, ast.Not,
    ast.Eq, ast.NotEq,
    ast.Compare, ast.Name, ast.Load, ast.Constant,
    ast.Subscript, ast.Dict, ast.Tuple, ast.List,
    ast.Attribute,
}
# Back-compat nodes added conditionally for Python version compatibility
if hasattr(ast, "Index"):
    ALLOWED_NODES.add(ast.Index)
if hasattr(ast, "Str"):
    ALLOWED_NODES.add(ast.Str)
if hasattr(ast, "Num"):
    ALLOWED_NODES.add(ast.Num)

class SafeExpressionError(Exception):
    """Custom exception for errors during safe expression evaluation."""
    pass

def _check_node(node: ast.AST) -> None:
    """Walk the AST and raise if any disallowed node types are found."""
    for child in ast.walk(node):
        if type(child) not in ALLOWED_NODES:
            raise SafeExpressionError(f"Disallowed expression node: {type(child).__name__}")

def _get_from_context(ctx: Dict[str, Any], name: str) -> Any:
    """Safely retrieve a value from the provided context."""
    return ctx.get(name)

def _evaluate_node(node: ast.AST, context: Dict[str, Any]) -> Any:
    """Recursively and safely evaluate an AST node."""
    if isinstance(node, ast.Expression):
        return _evaluate_node(node.body, context)

    if isinstance(node, ast.Constant):  # numbers, strings, bools, None
        return node.value
    # Back-compat for older Python versions
    if hasattr(ast, 'Num') and isinstance(node, ast.Num):
        return node.n
    if hasattr(ast, 'Str') and isinstance(node, ast.Str):
        return node.s

    if isinstance(node, ast.Name):
        return _get_from_context(context, node.id)

    if isinstance(node, ast.Attribute):
        base_obj = _evaluate_node(node.value, context)
        # Prevent access to private/special methods
        if node.attr.startswith('_'):
            raise SafeExpressionError("Access to private attributes is not allowed.")
        return getattr(base_obj, node.attr, None) if base_obj is not None else None

    if isinstance(node, ast.Subscript):
        base_obj = _evaluate_node(node.value, context)
        # Python 3.9+: slice is the key expression; older: ast.Index
        key = _evaluate_node(getattr(node.slice, "value", node.slice), context)
        if isinstance(base_obj, (dict, list, tuple)):
            try:
                return base_obj.get(key) if isinstance(base_obj, dict) else base_obj[key]
            except (KeyError, IndexError, TypeError):
                return None
        return None

    if hasattr(ast, 'Index') and isinstance(node, ast.Index):  # For Python < 3.9
        return _evaluate_node(node.value, context)

    if isinstance(node, ast.Compare):
        left = _evaluate_node(node.left, context)
        for op, comp_node in zip(node.ops, node.comparators):
            right = _evaluate_node(comp_node, context)
            if isinstance(op, ast.Eq):
                if not (left == right):
                    return False
            elif isinstance(op, ast.NotEq):
                if not (left != right):
                    return False
            else:
                raise SafeExpressionError("Only '==' and '!=' comparisons are supported.")
            left = right
        return True

    if isinstance(node, ast.BoolOp):
        values = [_evaluate_node(v, context) for v in node.values]
        if isinstance(node.op, ast.And):
            return all(values)
        if isinstance(node.op, ast.Or):
            return any(values)

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return not _evaluate_node(node.operand, context)

    raise SafeExpressionError(f"Unsupported expression node: {ast.dump(node)}")

def evaluate(expression: str, context: Dict[str, Any]) -> bool:
    """
    Safely evaluate a boolean expression string against a given context.
    """
    if not expression:
        return True
    try:
        tree = ast.parse(expression, mode="eval")
        _check_node(tree)
        result = _evaluate_node(tree, context)
        return bool(result)
    except (SyntaxError, SafeExpressionError, AttributeError, KeyError) as e:
        raise SafeExpressionError(f"Failed to safely evaluate expression '{expression}': {e}") from e