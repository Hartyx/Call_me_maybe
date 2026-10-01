"""Helpers for constrained decoding of string values."""

from llm_sdk import Small_LLM_Model


def generate_string_value(
    model: Small_LLM_Model,
    ids_list: list[int],
    id_to_token: dict[int, str]
) -> str:
    """Generate a string value using constrained token decoding.

    Starts with an opening double quote, generates tokens until a closing
    double quote is produced, then returns the complete quoted string.

    Args:
        model: The language model used to generate the string.
        ids_list: The token IDs representing the current model input.
        id_to_token: A mapping from token IDs to their corresponding text.

    Returns:
        The generated string value, including its surrounding double quotes.
    """
    # 1. ajouter le " ouvrant directement aux ids
    opening_ids = model.encode('"')[0].tolist()
    ids_list.extend(opening_ids)
    already_written = '"'

    while True:
        # 2. demander les scores au modele
        logits = model.get_logits_from_input_ids(ids_list)

        # 3. filtrer les tokens invalides
        filtered_logits = list(logits)
        for token_id in range(len(logits)):
            token_text = id_to_token.get(token_id)
            if token_text is None:
                filtered_logits[token_id] = float('-inf')
                continue
            # la concatenation doit toujours commencer par "
            if not (already_written + token_text).startswith('"'):
                filtered_logits[token_id] = float('-inf')

        # 4. choisir le meilleur token valide
        best_id = max(
            range(len(filtered_logits)),
            key=lambda i: filtered_logits[i]
        )
        token_text = id_to_token[best_id]
        already_written += token_text
        ids_list.append(best_id)

        # 5. verifier si la string est fermee (2eme guillemet)
        if already_written.count('"') >= 2:
            index_fermeture = already_written.index('"', 1)
            already_written = already_written[:index_fermeture + 1]
            break

    # 6. remplacer le marqueur d'espace du tokenizer
    already_written = already_written.replace("Ġ", " ")

    return already_written