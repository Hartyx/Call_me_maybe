from llm_sdk import Small_LLM_Model


def generate_number_value(
    model: Small_LLM_Model, ids_list: list[int], id_to_token: dict[int, str]
) -> str:
    """Generate a valid number value using constrained decoding.

    Args:
        model: The LLM model instance.
        ids_list: Current list of token IDs.
        id_to_token: Dictionary mapping token IDs to their text.

    Returns:
        A string representing a valid number.
    """
    already_written = ""
    limite = 10

    while len(already_written) < limite:
        logits = model.get_logits_from_input_ids(ids_list)

        raw_best_id = max(range(len(logits)), key=lambda i: logits[i])
        raw_best_text = id_to_token.get(raw_best_id, "")
        if already_written and (
            raw_best_text.startswith(",") or raw_best_text.startswith("}")
        ):
            break

        filtered_logits = list(logits)
        for token_id in range(len(logits)):
            token_text = id_to_token.get(token_id, "")
            concate = already_written + token_text

            est_valide = False
            try:
                float(concate)
                est_valide = True
            except ValueError:
                try:
                    float(concate + "0")
                    est_valide = True
                except ValueError:
                    est_valide = False

            if not est_valide:
                filtered_logits[token_id] = float("-inf")

        best_id = max(
            range(len(filtered_logits)),
            key=lambda i: filtered_logits[i]
            )
        token_text = id_to_token.get(best_id, "")

        already_written += token_text
        ids_list.append(best_id)

    return already_written
