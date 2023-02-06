from typing import Optional

from pydantic import BaseModel


class Question(BaseModel):
    text: str


class Entity(BaseModel):
    idx: str
    label: Optional[str]
    instance_of: Optional[str]
    description: Optional[list[str]]
    image: Optional[list[str]]


class PipelineResponce(BaseModel):
    answers: list[str]
