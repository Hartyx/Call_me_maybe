"""Utilities for constrained decoding of function parameters."""

from llm_sdk import Small_LLM_Model
from ..functions import FunctionDefinition
from ..parser import parser_functions


def generate_parameters(model: Small_LLM_Model, prompt: str, function_definition: FunctionDefinition) -> dict[str, str]:
    """Generate the parameters and their values for a function call."""
    result = {}

    input_ids = model.encode(prompt)[0].tolist()
    print(input_ids)
    for parameter_name in function_definition.parameters:
        parameter_type = function_definition.parameters[parameter_name].type
        value_tokens = []
        while True:
            logits = model.get_logits_from_input_ids(input_ids)
            allowed = set()
            if parameter_type == "number":
                for digit in "0123456789":
                    tokens = model.encode(digit)[0].tolist()
                    allowed.add(tokens[0])




if __name__ == "__main__":
    model = Small_LLM_Model()
    prompt = "bonjour tout le monde"

    input_ids = model.encode(prompt)[0].tolist()

    print(input_ids)