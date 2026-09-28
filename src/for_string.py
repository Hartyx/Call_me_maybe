from llm_sdk import Small_LLM_Model


def generate_string_value(
        model: Small_LLM_Model,
        ids_list: list[int],
        id_to_token: dict[int, str]
    ) -> str:
    """Generate a valid string value using constrained decoding.

    Args:
        model: The LLM model instance.
        ids_list: Current list of token IDs.
        id_to_token: Dictionary mapping token IDs to their text.

    Returns:
        A valid quoted string value.
    """
    already_written = ""
    limite = 50

    while len(already_written) < limite:
        # 1. demander les scores au modele
        logits = model.get_logits_from_input_ids(ids_list)

        # 2. filtrer les tokens invalides
        filtered_logits = list(logits)
        for token_id in range(len(logits)):
            token_text = id_to_token.get(token_id, "")
            concate = already_written + token_text

            est_valide = False

            # cas 1 : on n'a rien ecrit → doit commencer par "
            if already_written == "":
                est_valide = concate.startswith('"')

            # cas 2 : on a deja le " ouvrant → tout token est valide
            # sauf si on ferme avec " → on accepte pour terminer
            elif already_written.startswith('"'):
                est_valide = True

            if not est_valide:
                filtered_logits[token_id] = float('-inf')

        # 3. choisir le meilleur token valide
        best_id = max(range(len(filtered_logits)), key=lambda i: filtered_logits[i])
        token_text = id_to_token.get(best_id, "")

        # 4. ajouter le token
        already_written += token_text
        ids_list.append(best_id)

        # 5. verifier si la string est fermee
        # une string est complete si elle commence et finit par "
        if (len(already_written) > 1
                and already_written.startswith('"')
                and already_written.endswith('"')):
            break

    return already_written