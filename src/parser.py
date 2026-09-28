"""Utilities for loading and parsing JSON input files."""

from src.models import FunctionDefinition, Prompt
from pydantic import ValidationError
import json


def opening_file(file: str):
    """Load and parse a JSON file.

    Args:
        file: Path to the JSON file.

    Returns:
        The data loaded from the JSON file.

    Raises:
        ValueError: If the file does not exist or contains invalid JSON.
    """
    try:
        with open(file) as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        raise ValueError(f"File not found: {file}")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON in file: {file}")


def parser_json(file: str) -> list[FunctionDefinition]:
    """Parse function definitions from a JSON file.

    Each JSON object is validated as a FunctionDefinition.

    Args:
        file: Path to the function definitions file.

    Returns:
        A list of valid FunctionDefinition objects.
    """
    data = opening_file(file)
    result = []

    for element in data:
        try:
            func_def = FunctionDefinition(**element)
            result.append(func_def)
        except ValidationError as e:
            print(e)

    return result


def parser_prompts(file: str) -> list[Prompt]:
    """Parse prompts from a JSON file.

    Each JSON object is validated as a Prompt.

    Args:
        file: Path to the prompts file.

    Returns:
        A list of valid Prompt objects.
    """
    data = opening_file(file)
    result = []

    for element in data:
        try:
            prompt = Prompt(**element)
            result.append(prompt)
        except ValidationError as e:
            print(e)

    return result
