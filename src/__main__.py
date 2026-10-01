"""Process prompts and generate constrained function calls."""

from llm_sdk import Small_LLM_Model
import json
import argparse
from src import parser_prompts, parser_json, Result
from src.generator import generate_function_call
from pathlib import Path


def process_all_prompts(
    functions_path: str,
    tests_path: str,
    output_path: str,
) -> None:
    """Process all prompts and save the generated function calls.

    Loads the available functions and test prompts, generates a constrained
    function call for each prompt, and saves the results as JSON.

    Args:
        functions_path: Path to the function definitions JSON file.
        tests_path: Path to the test prompts JSON file.
        output_path: Path where the results will be saved.
    """
    model = Small_LLM_Model()

    vocab_path = model.get_path_to_vocab_file()
    with open(vocab_path, encoding="utf-8") as f:
        vocab = json.load(f)
    id_to_token = {v: k for k, v in vocab.items()}

    tok_path = model.get_path_to_tokenizer_file()
    with open(tok_path, encoding="utf-8") as f:
        tok_data = json.load(f)
    for tok in tok_data.get("added_tokens", []):
        id_to_token[tok["id"]] = tok["content"]

    all_functions = parser_json(functions_path)
    all_prompts = parser_prompts(tests_path)

    result = []

    for prompt_entry in all_prompts:
        question = prompt_entry.prompt

        functions_json = json.dumps([f.model_dump() for f in all_functions])

        guide = "\n".join(
            [f"- use {func.name} for: {func.description}" for func in all_functions]
        )

        texte_complet = (
            "Available functions: "
            f"{functions_json}\n\n"
            f"Question: {question}\n"
            f"Function guide:\n{guide}\n"
            f"Answer with a JSON object: "
        )

        ids_tensor = model.encode(texte_complet)
        ids_list = ids_tensor[0].tolist()

        genere = generate_function_call(model, ids_list, all_functions, id_to_token)

        parse = json.loads(genere)

        res = Result(
            prompt=question, name=parse["name"], parameters=parse["parameters"]
        )

        result.append(res)

    output_data = [r.model_dump() for r in result]
    texte = json.dumps(output_data, indent=2)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(texte)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate function calls from natural language prompts."
    )
    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json",
        help="Path to the functions definition JSON file.",
    )
    parser.add_argument(
        "--input",
        default="data/input/function_calling_tests.json",
        help="Path to the input prompts JSON file.",
    )
    parser.add_argument(
        "--output",
        default="data/output/function_calling_results.json",
        help="Path to the output JSON file.",
    )
    args = parser.parse_args()

    process_all_prompts(args.functions_definition, args.input, args.output)
