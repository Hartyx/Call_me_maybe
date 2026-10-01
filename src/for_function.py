from llm_sdk import Small_LLM_Model


def choose_next_token(
        logits: list[float],
        already_written: str,
        all_names: list[str],
        id_to_token: dict[int, str],
    ) -> int:
    """Choose the best valid token for function name generation.

    Args:
        logits: Score list from the LLM model.
        already_written: Text generated so far.
        all_names: List of valid function/value names.
        id_to_token: Dictionary mapping token IDs to their text.

    Returns:
        The ID of the best valid token.
    """
    filtered_logits = list(logits)

    for token_id in range(len(logits)):
        token_text = id_to_token.get(token_id)

        if token_text is None:
            filtered_logits[token_id] = float('-inf')
            continue

        concate = already_written + token_text
        remaining = [name for name in all_names if name.startswith(concate)]
        if len(remaining) == 0:
            filtered_logits[token_id] = float('-inf')

    best_id = max(range(len(filtered_logits)), key=lambda i: filtered_logits[i])
    return best_id


def get_fullname_func(
        model: Small_LLM_Model,
        ids_list: list[int],
        all_names: list[str],
        id_to_token: dict[int, str]
    ) -> str:
    """Generate a valid complete name using constrained decoding.

    Args:
        model: The LLM model instance.
        ids_list: Current list of token IDs.
        all_names: List of valid names to choose from.
        id_to_token: Dictionary mapping token IDs to their text.

    Returns:
        A complete valid name from all_names.
    """
    already_written = ""

    while True:
        logits = model.get_logits_from_input_ids(ids_list)

        best_id = choose_next_token(logits, already_written, all_names, id_to_token)
        token_text = id_to_token[best_id]

        already_written += token_text
        ids_list.append(best_id)

        if already_written in all_names:
            break

    return already_written
