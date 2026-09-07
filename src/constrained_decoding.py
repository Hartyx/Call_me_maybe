"""Constrained decoding for functions."""
from llm_sdk import Small_LLM_Model
from .functions import FunctionDefinition
from .parser import parser_functions
import math
import numpy as np

def build_function_name_tokens(model: Small_LLM_Model, function_definitions: list[FunctionDefinition]) -> dict[str, list[int]]:
    result = {}
    for functions in function_definitions:
        tensor_tokens = model.encode(functions.name)
        tokens = tensor_tokens[0].tolist()
        result[functions.name] = tokens
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


def generate_function_name(model: Small_LLM_Model, prompt: str, function_name_tokens: str) -> dict[str, list[int]]:
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



if __name__ == "__main__":
    model = Small_LLM_Model()
    functions = parser_functions('data/input/functions_definition.json')
    tokens = build_function_name_tokens(model, functions)
    prompt = "What is the sum of 2 and 3?"
    name = generate_function_name(model, prompt, tokens)
    print(f"la fonction est: {name}")
