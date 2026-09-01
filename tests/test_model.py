from llm_sdk import Small_LLM_Model

if __name__ == "__main__":
    model = Small_LLM_Model()
    prompt = "What is the sum of 2 and 3?"
    input_ids = model.encode(prompt)
    print(f"\nprompt: {prompt}")
    print(f"Token IDS: {input_ids}")

    logits = model.get_logits_from_input_ids(input_ids[0].tolist())
    print(f"\nNombres de logits: {len(logits)}\n")

    best_token_id = max(range(len(logits)), key=lambda i: logits[i])
    print("\nMeilleur token ID :", best_token_id)
    print(f"Meilleur token : '{model.decode([best_token_id])}'") 
    print("Meilleur logit :", logits[best_token_id])
    print()

    function_name = "fn_add_numbers"

    function_ids = model.encode(function_name)
    print()
    print("Fonction :", function_name)
    print("Token IDs :", function_ids)
    print("Décodage :", model.decode(function_ids))
    print()
    # print("\n -------------------------------------")
    functions = [
    "fn_add_numbers",
    "fn_greet",
    "fn_reverse_string",
    "fn_get_square_root",
    "fn_substitute_string_with_regex",
    ]

    for function in functions:
        ids = model.encode(function)

        print(function)
        print(ids)
        print()