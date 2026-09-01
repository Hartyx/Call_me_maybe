from pydantic import BaseModel, ValidationError
import json
class FunctionCallingTest(BaseModel):
    prompt: str

def load(path_file: str) -> list[FunctionCallingTest]:
    try:
        with open(path_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print("Error")
        return []
    except json.JSONDecodeError:
        print("Error")
        return []

def function_calling(path_file):
    raw_data = load(path_file)
    result = []
    for element in raw_data:
        try:
            fn_calling = FunctionCallingTest(**element)
            result.append(fn_calling)
        except ValidationError as e:
            print(f"Error: {e}")
    return result

if __name__ == "__main__":
    result = function_calling('data/input/function_calling_tests.json')
    for test in result:
        print(test.prompt)
    
