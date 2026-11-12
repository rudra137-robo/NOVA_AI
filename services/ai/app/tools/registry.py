
import ast
import operator
from datetime import datetime

from app.research.web_search import search_web


# A deliberately limited calculator.
# It never executes arbitrary Python code.

_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
}

_UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _calculate_node(node):
    if isinstance(node, ast.Constant):
        if type(node.value) in (int, float):
            if abs(node.value) > 1e100:
                raise ValueError("Number is too large.")
            return node.value
        raise ValueError("Only numeric values are allowed.")

    if isinstance(node, ast.BinOp):
        if type(node.op) is ast.Pow:
            left = _calculate_node(node.left)
            right = _calculate_node(node.right)

            if abs(right) > 8:
                raise ValueError("Exponent is limited to 8.")

            if left == 0 and right < 0:
                raise ValueError("Division by zero.")

            return left ** right

        operation = _OPERATORS.get(type(node.op))
        if operation is None:
            raise ValueError("Unsupported operator.")

        left = _calculate_node(node.left)
        right = _calculate_node(node.right)

        if type(node.op) in (ast.Div, ast.FloorDiv, ast.Mod):
            if right == 0:
                raise ValueError("Division by zero.")

        return operation(left, right)

    if isinstance(node, ast.UnaryOp):
        operation = _UNARY_OPERATORS.get(type(node.op))
        if operation is None:
            raise ValueError("Unsupported unary operator.")
        return operation(_calculate_node(node.operand))

    raise ValueError("Unsupported expression.")


def calculate(expression):
    if not isinstance(expression, str):
        raise ValueError("Expression must be text.")

    if not expression or len(expression) > 100:
        raise ValueError("Expression must contain 1–100 characters.")

    tree = ast.parse(expression, mode="eval")
    result = _calculate_node(tree.body)

    if isinstance(result, complex):
        raise ValueError("Complex results are not supported.")

    if isinstance(result, float):
        if not __import__("math").isfinite(result):
            raise ValueError("Result is not finite.")

    return str(result)


def get_current_time():
    return datetime.now().astimezone().isoformat(
        timespec="seconds"
    )


TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": (
                "Evaluate a mathematical expression when an accurate "
                "calculation is needed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": (
                            "A mathematical expression, e.g. (25 * 4) + 10"
                        ),
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": (
                "Get the current date and local time of the computer "
                "running NOVA."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": (
                "Search the web for current information, recent events, "
                "or facts that need external sources. Use when relevant. "
                "Do not claim to have searched if this tool was not used."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A concise web search query."
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Number of results, from 1 to 5."
                    }
                },
                "required": ["query"],
            },
        },
    },
]



def execute_tool(name, arguments):
    if not isinstance(arguments, dict):
        return "Tool error: arguments must be an object."

    try:
        if name == "calculate":
            return calculate(arguments.get("expression"))

        if name == "get_current_time":
            return get_current_time()

        if name == "search_web":
            query = arguments.get("query", "")
            max_results = arguments.get("max_results", 5)

            if not isinstance(query, str) or not query.strip():
                raise ValueError("A non-empty search query is required.")

            if (
                not isinstance(max_results, int)
                or isinstance(max_results, bool)
                or not 1 <= max_results <= 5
            ):
                raise ValueError("max_results must be between 1 and 5.")

            return search_web(query, max_results)

        return f"Tool error: '{name}' is not an approved tool."

    except (ValueError, TypeError, SyntaxError, OverflowError,
            ZeroDivisionError) as exc:
        return f"Tool error: {exc}"

   