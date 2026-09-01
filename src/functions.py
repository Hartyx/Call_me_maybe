"""Data models representing callable function definitions.

This module defines the pydantic models used to parse and validate
the function definitions provided in ``function_definitions.json``.
"""

from pydantic import BaseModel
from typing import Literal


class Parameter(BaseModel):
    """Define a parameter type."""

    type: Literal["string", "number", "boolean"]


class FunctionDefinition(BaseModel):
    """Define a callable function."""

    name: str
    description: str
    parameters: dict[str, Parameter]
    returns: Parameter
