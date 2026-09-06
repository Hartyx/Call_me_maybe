import json
from .functions import FunctionDefinition 
from pydantic import ValidationError


def load_json(path_file: str) -> dict | None:
    try:
        with open(path_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print("Error: File not Found")
        return
    except json.JSONDecodeError:
        print("Error at the json file")
        return


def  parser_functions(path_file: str) -> list[FunctionDefinition]:
    raw_data = load_json(path_file)
    result = []
    if raw_data is None:
        raise ValueError("the json file is empty")
    for element in raw_data:
        try:
            function_definition = FunctionDefinition(**element)
            result.append(function_definition)
        except ValidationError as e:
            print(f"Error: {e}")
    return result


if __name__ == "__main__":
    result = parser_functions('data/input/function_definition.json')
    print(f"Nombre de fonctions chargee: {len(result)}")
    for f in result:
        print(f.name)
