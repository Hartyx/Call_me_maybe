"""Entry point for the function calling generation pipeline."""

from llm_sdk import Small_LLM_Model
import json
import argparse
from pathlib import Path
from src import parser_prompts, parser_json, Result
from src.generator import generate_function_call


def process_all_prompts(
    functions_path: str,
    tests_path: str,
    output_path: str,
) -> None:
    """Process all prompts and generate function calls.

    Args:
        functions_path: Path to functions_definition.json.
        tests_path: Path to function_calling_tests.json.
        output_path: Path to write the output JSON file.
    """
    # 1. charger le modele
    model = Small_LLM_Model()

    # 2. charger le vocabulaire
    vocab_path = model.get_path_to_vocab_file()
    with open(vocab_path, encoding="utf-8") as f:
        vocab = json.load(f)
    id_to_token = {v: k for k, v in vocab.items()}

    tok_path = model.get_path_to_tokenizer_file()
    with open(tok_path, encoding="utf-8") as f:
        tok_data = json.load(f)
    for tok in tok_data.get("added_tokens", []):
        id_to_token[tok["id"]] = tok["content"]

    # 3. charger les fonctions et les prompts
    all_functions = parser_json(functions_path)
    all_prompts = parser_prompts(tests_path)

    # 4. generer les resultats
    result = []
    for prompt_entry in all_prompts:
        try:
            question = prompt_entry.prompt
            functions_json = json.dumps([f.model_dump() for f in all_functions])

            texte_complet = (
                f"Available functions: {functions_json}\n\n"
                f"Question: {question}\n"
                f"Select the most appropriate function and extract "
                f"the exact values from the question as parameters.\n"
                f"Answer with a JSON object: "
            )

            ids_list = model.encode(texte_complet)[0].tolist()
            genere = generate_function_call(
                model,
                ids_list,
                all_functions,
                id_to_token
            )
            parse = json.loads(genere)
            res = Result(
                prompt=question,
                name=parse["name"],
                parameters=parse["parameters"]
            )
            result.append(res)

        except Exception as e:
            print(f"ERROR processing prompt '{prompt_entry.prompt}': {e}")

    # 5. ecrire le fichier output
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    output_data = [r.model_dump() for r in result]
    texte = json.dumps(output_data, indent=2)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(texte)

    print(f"Output written to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate function calls from natural language prompts."
    )
    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json",
        help="Path to the functions definition JSON file."
    )
    parser.add_argument(
        "--input",
        default="data/input/function_calling_tests.json",
        help="Path to the input prompts JSON file."
    )
    parser.add_argument(
        "--output",
        default="data/output/function_calling_results.json",
        help="Path to the output JSON file."
    )
    args = parser.parse_args()

    process_all_prompts(
        args.functions_definition,
        args.input,
        args.output
    )