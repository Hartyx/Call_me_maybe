"""Constrained decoding utilities."""
from llm_sdk import Small_LLM_Model
from ..functions import FunctionDefinition
from ..parser import parser_functions


def build_function_name_tokens(model, function_definitions: list[FunctionDefinition]) -> dict[str, list[int]]:
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


def generate_function_name(model, prompt: str, function_name_tokens: dict[str, list[int]]) -> str:
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
        best_token = max(allowed, key=lambda token_id: logits[token_id])
        generated.append(best_token)
    name_choose = is_function_name_complete(function_name_tokens, generated)
    return name_choose


if __name__ == "__main__":
    functions = parser_functions("data/input/functions_definition.json")
    model = Small_LLM_Model()
    function_name_tokens = build_function_name_tokens(model, functions)

    prompt = "Greet shrek"
    result = generate_function_name(model, prompt, function_name_tokens)
    print(f"\nFonction Name: {result}\n")

    prompts_test = [
    "What is the sum of 2 and 3?",
    "Greet shrek",
    "Reverse the string 'hello'",
]
    for p in prompts_test:
        result = generate_function_name(model, p, function_name_tokens)
        print(f"{p} -> {result}")

