"""calculator: safe Decimal arithmetic over an AST whitelist.

Allowed: numeric literals, unary +/-, and + - * / // %. Nothing else parses into a result: no
names, calls, attributes, subscripts, comparisons, and no exponentiation (the denial-of-service
vector). Evaluation runs in a local Decimal context with 40 significant digits; the global
context used elsewhere in the package is never touched.
"""

from __future__ import annotations

import ast
import operator
from collections.abc import Callable
from decimal import Decimal, InvalidOperation, localcontext
from typing import Final

from fintrace.constants import MAX_CALCULATOR_EXPR_CHARS
from fintrace.environment.observations import build_error_result, build_ok_result
from fintrace.models import CalculatorArgs, EpisodeState, JsonObject, PublicEpisode, ToolResult

_PRECISION: Final[int] = 40
_MAX_EXPONENT: Final[int] = 40  # |value| must stay below ten to the 40th (and above its inverse)
_BINARY_OPS: Final[dict[type[ast.operator], Callable[[Decimal, Decimal], Decimal]]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
}


class CalcError(ValueError):
    """Internal reason code (TOO_LONG, FORBIDDEN, SYNTAX_ERROR, DIV_ZERO, NOT_FINITE)."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


def _literal(node: ast.Constant, source: str) -> Decimal:
    """Exact Decimal from the literal's own source text (never through a float round-trip)."""
    value = node.value
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise CalcError("FORBIDDEN")
    text = ast.get_source_segment(source, node)
    if text is None:
        raise CalcError("FORBIDDEN")
    try:
        literal = Decimal(text.replace("_", ""))
    except InvalidOperation as exc:  # hex / octal / binary spellings are not money
        raise CalcError("FORBIDDEN") from exc
    if not literal.is_finite():
        raise CalcError("NOT_FINITE")
    return literal


def _evaluate(node: ast.AST, source: str) -> Decimal:
    if isinstance(node, ast.Expression):
        return _evaluate(node.body, source)
    if isinstance(node, ast.Constant):
        return _literal(node, source)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd | ast.USub):
        operand = _evaluate(node.operand, source)
        return operand if isinstance(node.op, ast.UAdd) else -operand
    if isinstance(node, ast.BinOp):
        binary = _BINARY_OPS.get(type(node.op))
        if binary is None:
            raise CalcError("FORBIDDEN")
        return binary(_evaluate(node.left, source), _evaluate(node.right, source))
    raise CalcError("FORBIDDEN")


def safe_eval(expression: str) -> Decimal:
    """Evaluate ``expression`` as exact Decimal arithmetic; raise CalcError for anything else."""
    if len(expression) > MAX_CALCULATOR_EXPR_CHARS:
        raise CalcError("TOO_LONG")
    try:
        tree = ast.parse(expression, mode="eval")
    except (SyntaxError, ValueError, RecursionError, MemoryError) as exc:
        raise CalcError("SYNTAX_ERROR") from exc
    try:
        with localcontext() as context:
            context.prec = _PRECISION
            result = _evaluate(tree, expression)
    except ZeroDivisionError as exc:
        raise CalcError("DIV_ZERO") from exc
    except (InvalidOperation, OverflowError, RecursionError) as exc:
        raise CalcError("NOT_FINITE") from exc
    if not result.is_finite() or (result != 0 and abs(result.adjusted()) > _MAX_EXPONENT):
        raise CalcError("NOT_FINITE")
    return result


def render(value: Decimal) -> str:
    """Plain positional text (no exponent), trailing zeros kept: ``135525.0000``."""
    return format(value, "f")


def handle_calculator(
    args: CalculatorArgs, state: EpisodeState, episode: PublicEpisode
) -> ToolResult:
    """``{value: str}`` (Decimal as text for byte stability); any failure -> CALC_ERROR."""
    try:
        value = safe_eval(args.expression)
    except CalcError as exc:
        return build_error_result("calculator", "CALC_ERROR", exc.reason, state)
    result: JsonObject = {"value": render(value)}
    return build_ok_result("calculator", result, state)


__all__ = ["CalcError", "handle_calculator", "render", "safe_eval"]
