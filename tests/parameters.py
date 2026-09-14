"""Utilities for constrained decoding of function parameters."""

from llm_sdk import Small_LLM_Model
from ..functions import FunctionDefinition
from ..parser import parser_functions


def get_parameter_names(function_definition: FunctionDefinition) -> list[str]:
    """Return the names of all parameters defined for a function."""
    return list(function_definition.parameters.keys())


def get_parameter_type(function_definition: FunctionDefinition, parameter_name: str) -> str:
    """Return the type of a given parameter."""
    return function_definition.parameters[parameter_name].type


def get_separator_tokens(model: Small_LLM_Model, function_definition: FunctionDefinition, parameter_name: str) -> set[int]:
    """Return the token allowed to end the current value (comma or closing brace)."""
    names = get_parameter_names(function_definition)
    is_last = names.index(parameter_name) == len(names) - 1
    text = "}" if is_last else ","
    return {model.encode(text)[0].tolist()[0]}


def get_allowed_value_tokens(model: Small_LLM_Model, parameter_type: str) -> set[int]:
    """Return the tokens allowed to start/continue a value, based on its type."""
    result = set()
    if parameter_type == "number":
        for digit in "0123456789":
            result.add(model.encode(digit)[0].tolist()[0])
    elif parameter_type == "boolean":
        result.add(model.encode("true")[0].tolist()[0])
        result.add(model.encode("false")[0].tolist()[0])
    return result


def select_best_allowed_token(logits: list[float], allowed_tokens: set[int]) -> int:
    """Return the allowed token with the highest logit score."""
    best_token = None
    best_logit = -math.inf
    for token in allowed_tokens:
        if logits[token] > best_logit:
            best_logit = logits[token]
            best_token = token
    return best_token


def generate_parameters(model: Small_LLM_Model, prompt: str, function_definition: FunctionDefinition) -> dict[str, str]:
    """Generate the parameters and their values for a function call."""
    result = {}
    input_ids = model.encode(prompt)[0].tolist()

    for parameter_name in get_parameter_names(function_definition):
        parameter_type = get_parameter_type(function_definition, parameter_name)
        allowed_values = get_allowed_value_tokens(model, parameter_type)
        separator = get_separator_tokens(model, function_definition, parameter_name)

        value_tokens = []
        while True:
            logits = model.get_logits_from_input_ids(input_ids)
            best_token = select_best_allowed_token(logits, allowed_values | separator)

            if best_token in separator:
                break

            value_tokens.append(best_token)
            input_ids.append(best_token)

        result[parameter_name] = model.decode(value_tokens)

    return result


if __name__ == "__main__":
    functions = parser_functions("data/input/function_definitions.json")
    model = Small_LLM_Model()
    function_definition = functions[0]
    prompt = "What is the sum of 2 and 3?"
    result = generate_parameters(model, prompt, function_definition)
    print(f"parameters: {result}")


def generate_parameters(model: Small_LLM_Model, prompt: str, function_definition: FunctionDefinition) -> dict[str, str]:
    """Generate the parameters and their values for a function call."""

    # Étape 1 : préparer un dictionnaire vide pour stocker les résultats
    result = {}

    # Étape 2 : transformer le prompt en tokens de départ
    # (rappelez-vous : model.encode(...) retourne un tensor 2D, il faut l'aplatir)
    input_ids = ...

    # Étape 3 : boucler sur chaque paramètre attendu par la fonction
    # (function_definition.parameters est un dictionnaire {nom: Parameter})
    for parameter_name in ...:

        # Étape 4 : récupérer le TYPE de ce paramètre (number, string, boolean)
        parameter_type = ...

        # Étape 5 : préparer une liste vide pour accumuler les tokens de la valeur
        value_tokens = []

        # Étape 6 : boucle interne, tant qu'on n'a pas fini d'écrire la valeur
        while True:

            # Étape 6a : demander les scores au modèle
            logits = ...

            # Étape 6b : déterminer quels tokens sont autorisés MAINTENANT
            # (dépend du type : des chiffres si "number", true/false si "boolean"...)
            # PLUS les tokens de séparation (virgule ou accolade fermante)
            allowed = ...

            # Étape 6c : choisir le meilleur token autorisé
            best_token = ...

            # Étape 6d : si c'est un séparateur (virgule/accolade), on arrête cette valeur
            if ...:
                break

            # Étape 6e : sinon, on ajoute ce token à la valeur en construction
            value_tokens.append(best_token)
            input_ids.append(best_token)

        # Étape 7 : convertir les tokens accumulés en texte, et stocker le résultat
        result[parameter_name] = ...

    # Étape 8 : retourner le dictionnaire final
    return result