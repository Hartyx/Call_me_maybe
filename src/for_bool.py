from src.for_function import get_fullname_func
from llm_sdk import Small_LLM_Model


def generate_bool_value(
    model: Small_LLM_Model, ids_list: list[int], id_to_token: dict[int, str]
) -> str:
    """Generate a boolean value ('true' or 'false') using the LLM.

    Args:
        model: The LLM model used to generate tokens.
        ids_list: Token IDs representing the current input.
        id_to_token: Mapping from token IDs to token text.

    Returns:
        The generated boolean value as a string.
    """

    result = get_fullname_func(model, ids_list, ["true", "false"], id_to_token)
    return result
