
import pytest

from app.tools import calculate, get_current_time


def test_calculator_addition():
    assert calculate("25 + 10") == "35"


def test_calculator_multiplication():
    assert calculate("25 * 4 + 10") == "110"


def test_calculator_division():
    assert calculate("100 / 4") == "25.0"


def test_calculator_division_by_zero():
    with pytest.raises(ValueError, match="Division by zero"):
        calculate("10 / 0")


def test_calculator_rejects_python_code():
    with pytest.raises((ValueError, TypeError, SyntaxError)):
        calculate("__import__('os').system('whoami')")


def test_current_time_returns_valid_timestamp():
    from datetime import datetime

    timestamp = get_current_time()
    parsed = datetime.fromisoformat(timestamp)

    assert parsed is not None
