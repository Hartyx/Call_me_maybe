"""Constrained decoding utilities."""
import math
import numpy as np
from llm_sdk import Small_LLM_Model
from .functions import FunctionDefinition
from .parser import parser_functions
import json
from typing import Any


def build_function_name_tokens(model: Small_LLM_Model, function_definitions: list[FunctionDefinition]) -> dict[str, list[int]]:
    """Builds the mapping function_name → list of token_ids,
    using the model's ACTUAL tokens."""
    result = {}
    for function in function_definitions:
        tensor_tokens = model.encode(function.name)
        tokens = tensor_tokens[0].tolist()
        result[function.name] = tokens
    return result


def get_next_allowed_tokens(function_name_tokens: dict[str, list[int]], prefix: list[int]) -> set[int]:
    """Returns the set of valid tokens to continue with,
    based on what has already been generated."""
    allowed = set()
    for tokens in function_name_tokens.values():
        if tokens[:len(prefix)] == prefix and len(prefix) < len(tokens):
            allowed.add(tokens[len(prefix)])
    return allowed


def is_function_name_complete(function_name_tokens: dict[str, list[int]], prefix: list[int]) -> str | None:
    """Returns the function name if the prefix 
    exactly matches its complete tokens."""
    for name, tokens in function_name_tokens.items():
        if tokens == prefix:
            return name
    return None


def mask_logits(logits: list[float], allowed: set[int]) -> list[float]:
    new_logits = logits.copy()
    for index in range(len(logits)):
        if index not in allowed:
            new_logits[index] = -math.inf
    return new_logits

def generate_function_name(model: Small_LLM_Model, prompt: str, function_name_tokens: dict[str, list[int]]) -> str:
    """Generates the selected function name token by token,
    while respecting the constraints defined by function_name_tokens."""
    prompt_tokens = model.encode(prompt)
    prompt_tokens = prompt_tokens[0].tolist()

    generated = []

    while not is_function_name_complete(function_name_tokens, generated): 
        current_input = prompt_tokens + generated
        logits = model.get_logits_from_input_ids(current_input)
        allowed = get_next_allowed_tokens(function_name_tokens, generated)
        if not allowed:
            raise ValueError("Error")
        masked = mask_logits(logits, allowed)
        best_token = np.argmax(masked)
        generated.append(int(best_token))
    name_choose = is_function_name_complete(function_name_tokens, generated)
    return name_choose


def force_tokens(model: Small_LLM_Model, text: str) -> list[str]:
    """Encode a fixed, known text into tokens, without any model choice."""
    tensor = model.encode(text)
    return tensor[0].tolist()


def is_number_token(model: Small_LLM_Model, token_id: int) -> bool:
    """Check if a token's text is a valid part of number (digit or dot)."""
    text = model.decode([token_id])
    return text.strip() in [str(d) for d in range(10)]


def generate_number(model: Small_LLM_Model, base_tokens: list[int], generated_so_far: list[int]) -> tuple[list[int], float]:
    """Generates a number token by token. Returns the tokens generated and the parsed float value."""
    number_tokens = []
    number_text = ""

    while True:
        current_input = base_tokens + generated_so_far + number_tokens
        logits = model.get_logits_from_input_ids(current_input)

        allowed = set()
        for token_id in range(len(logits)):
            text = model.decode([token_id]).strip()
            if text in [str(d) for d in range(10)]:
                allowed.add(token_id)
            elif text in [",", "}"] and len(number_tokens) > 0:
                allowed.add(token_id)

        masked = mask_logits(logits, allowed)
        best_token = int(np.argmax(masked))
        best_text = model.decode([best_token]).strip()

        if best_text in [",", "}"]:
            break

        number_tokens.append(best_token)
        number_text += best_text

        if len(number_tokens) >= 5:
            break

    value = float(number_text)
    return number_tokens, value


def generate_arguments(
    model: Small_LLM_Model,
    prompt_tokens: list[int],
    name_tokens: list[int],
    function_definition: FunctionDefinition,
) -> dict[str, Any]:
    """Generates the JSON arguments for the chosen function, respecting its schema."""

    base_tokens = prompt_tokens + name_tokens
    generated_so_far = force_tokens(model, "{")

    remaining_params = list(function_definition.parameters.keys())
    result: dict[str, Any] = {}

    while remaining_params:
        param_name = remaining_params[0]
        param_def = function_definition.parameters[param_name]

        key_tokens = force_tokens(model, f'"{param_name}":')
        generated_so_far = generated_so_far + key_tokens

        if param_def.type == "number":
            value_tokens, value = generate_number(model, base_tokens, generated_so_far)
            generated_so_far = generated_so_far + value_tokens
            result[param_name] = value
        else:
            raise ValueError(f"Type not supported yet: {param_def.type}")

        remaining_params.pop(0)

        if remaining_params:
            comma_tokens = force_tokens(model, ",")
            generated_so_far = generated_so_far + comma_tokens

    closing_tokens = force_tokens(model, "}")
    generated_so_far = generated_so_far + closing_tokens

    return result


if __name__ == "__main__":
    functions = parser_functions("data/input/functions_definition.json")
    model = Small_LLM_Model()
    function_name_tokens = build_function_name_tokens(model, functions)

    prompt = "What is the sum of 2 and 3?"
    prompt_tokens = model.encode(prompt)
    prompt_tokens = prompt_tokens[0].tolist()

    name = generate_function_name(model, prompt, function_name_tokens)
    print("name:", name)

    function_def = next(f for f in functions if f.name == name)
    name_tokens = function_name_tokens[name]

    arguments = generate_arguments(model, prompt_tokens, name_tokens, function_def)
    print("parameters:", arguments)