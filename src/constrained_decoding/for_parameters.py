"""Utilities for constrained decoding of function parameters.

This module provides helper functions to extract parameter names,
convert them into model token IDs, and determine which parameter
tokens are allowed during constrained decoding.
"""

from llm_sdk import Small_LLM_Model
from ..functions import FunctionDefinition
from ..parser import parser_functions


def get_parameter_names(function_definition: FunctionDefinition) -> list[str]:
    """Return the names of all parameters defined for a function."""
    result = []

    for parameter in function_definition.parameters:
        result.append(parameter)

    return result


def build_parameter_name_tokens(
    model: Small_LLM_Model,
    function_definitions: FunctionDefinition
) -> dict[str, list[int]]:
    """Map each parameter name to its corresponding model token IDs."""
    result = {}

    for parameter in get_parameter_names(function_definitions):
        tensor_tokens = model.encode(parameter)
        tokens = tensor_tokens[0].tolist()
        result[parameter] = tokens

    return result


def get_allowed_parameter_tokens(
    parameter_name_tokens: dict[str, list[int]]
) -> set[int]:
    """Return the first token ID allowed for each parameter name."""
    result = set()

    for parameter in parameter_name_tokens.values():
        result.add(parameter[0])

    return result


def get_next_allowed_parameter_tokens(
    parameter_name_tokens:dict[str, list[int]],
    prefix: list[int]
    ) -> set[int]:

    allowed = set()

    for tokens in parameter_name_tokens.values():
        if tokens[:len(prefix)] == prefix and len(prefix) < len(tokens):
            allowed.add(tokens[len(prefix)])
    return allowed


if __name__ == "__main__":
    parameters = parser_functions("data/input/functions_definition.json")
    model = Small_LLM_Model()

    print(get_parameter_names(parameters[0]))
    print()
    print(build_parameter_name_tokens(model, parameters[0]))