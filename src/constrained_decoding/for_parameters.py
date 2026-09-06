from llm_sdk import Small_LLM_Model
from ..functions import FunctionDefinition
from ..parser import parser_functions


def get_parameter_names(function_definition: FunctionDefinition) -> list[str]:
    result = []
    for parameter in function_definition.parameters:
        result.append(parameter)
    return result
    
def build_parameter_name_tokens(model: Small_LLM_Model, function_definitions: FunctionDefinition) -> dict[str, list[int]]:
    result = {}
    for parameter in get_parameter_names(function_definitions):
        tensor_tokens = model.encode(parameter)
        tokens = tensor_tokens[0].tolist()
        result[parameter] = tokens
    return result

if __name__ ==  "__main__":
    parameters = parser_functions("data/input/functions_definition.json")
    model = Small_LLM_Model()

    names = ["name", "age", "city"]
    print(get_parameter_names(parameters[0]))
