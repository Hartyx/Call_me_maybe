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


def is_parameter_name_complete(parameter_name_tokens: dict[str, list[int]], prefix: list[int]) -> str | None:
    """Return the parameter name if the prefix exactly
    matches its full token sequence."""
    for name, tokens in parameter_name_tokens.items():
        if tokens == prefix:
            return name
    return None


def get_colon_token(model: Small_LLM_Model) -> int:
    """Return the token ID corresponding to the colon character."""
    tensor_tokens = model.encode(":")
    return tensor_tokens[0].tolist()[0]
        

def get_allowed_colon_tokens(colon_token: int) -> set[int]:
    """Return the set containing the colon token."""
    result = set()
    result.add(colon_token)
    return result


<<<<<<< HEAD
=======
def get_quote_token(model: Small_LLM_Model) -> int:
    """Return the token ID corresponding to the double quote character."""
    tensor_tokens = model.encode('"')
    return tensor_tokens[0].tolist()[0]


>>>>>>> fb6d7da (ajouter get_separator_tokens dans for_parameters.py)
def get_allowed_value_tokens(model: Small_LLM_Model, parameter_type: str) -> set[int]:
    """Return the token IDs allowed for a parameter value type."""
    result = set()
    if parameter_type == "boolean":
<<<<<<< HEAD
        tokens_true = model.encode("True")
        tokens_false = model.encode("False")
=======
        tokens_true = model.encode("true")
        tokens_false = model.encode("false")
>>>>>>> fb6d7da (ajouter get_separator_tokens dans for_parameters.py)
        result.add(tokens_true[0].tolist()[0])
        result.add( tokens_false[0].tolist()[0])
    if parameter_type == "number":
        for number in "0123456789":
            tokens_number = model.encode(number)
            result.add(tokens_number[0].tolist()[0])
<<<<<<< HEAD
    return result


if __name__ == "__main__":
    parameters = parser_functions("data/input/functions_definition.json")
    model = Small_LLM_Model()

    # print(get_parameter_names(parameters[0]))
    # print()
    # print(build_parameter_name_tokens(model, parameters[0]))

    print(f"ok: {get_colon_token(model)}\n")
    # print(f"{get_allowed_colon_tokens(15)}\n")
    print(get_allowed_value_tokens(model, "number"))
=======
    if parameter_type == "string":
        result.add(get_quote_token(model))
    return result


def get_closing_tokens(model: Small_LLM_Model) -> set[int]:
    """Return the token IDs corresponding to comma and closing brace."""
    result = set()
    tokens_comma = model.encode(",")
    tokens_closing = model.encode("}")
    result.add(tokens_comma[0].tolist()[0])
    result.add(tokens_closing[0].tolist()[0])
    return result


def get_parameter_type(
    function_definition: FunctionDefinition,
    parameter_name: str
) -> str:
    """Return the type of a parameter from its function definition."""
    parameter = function_definition.parameters[parameter_name]
    return parameter.type


def get_allowed_tokens_for_parameter(
    model: Small_LLM_Model,
    function_definition: FunctionDefinition,
    parameter_name: str
) -> set[int]:
    """Return the token IDs allowed for a parameter value."""
    parameter_type = get_parameter_type(function_definition, parameter_name)
    return get_allowed_value_tokens(model, parameter_type)

def has_more_parameters(
    function_definition: FunctionDefinition,
    parameter_name: str
) -> bool:
    """Return whether parameters remain after the given parameter."""
    parameter_names = get_parameter_names(function_definition)
    parameter_index = parameter_names.index(parameter_name)
    return parameter_index < len(parameter_names) - 1

def get_separator_tokens(
    model: Small_LLM_Model,
    function_definition: FunctionDefinition,
    parameter_name: str
) -> set[int]:
    """Return the token ID for comma or closing brace after a parameter."""
    has_more = has_more_parameters(function_definition, parameter_name)
    if has_more:
        tokens_comma = model.encode(",")
        return {tokens_comma[0].tolist()[0]}
    else:
        tokens_closing = model.encode("}")
        return {tokens_closing[0].tolist()[0]}

if __name__ == "__main__":
    parameters = parser_functions("data/input/functions_definition.json")
    model = Small_LLM_Model()
    # print(get_parameter_names(parameters[0]))
    # print()
    # print(build_parameter_name_tokens(model, parameters[0]))
    print(f"ok: {get_colon_token(model)}\n")
    # print(f"{get_allowed_colon_tokens(15)}\n")
    print(get_allowed_value_tokens(model, "number"))
    print(get_quote_token(model))

