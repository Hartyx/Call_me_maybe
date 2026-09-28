from typing import Literal
from pydantic import BaseModel


class ParameterFunction(BaseModel):
    """Define the type of a function parameter."""
    type: Literal["number", "string", "boolean"]


class FunctionReturn(BaseModel):
    """Define the return type of a function."""
    type: Literal["number", "string", "boolean"]


class FunctionDefinition(BaseModel):
    """Define a function that can be called by the model.

    Contains its name, description, parameters, and return type.
    """
    name: str
    description: str
    parameters: dict[str, ParameterFunction]
    returns: FunctionReturn


class Prompt(BaseModel):
    """Represent a natural-language prompt given to the model."""
    prompt: str


class Result(BaseModel):
    """Represent the generated function call result.

    Contains the original prompt, selected function name,
    and generated parameter values.
    """
    prompt: str
    name: str
    parameters: dict[str, int | float | str | bool]
