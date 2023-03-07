from typing import Optional

from pydantic import BaseModel


class Question(BaseModel):
    text: str


class Entity(BaseModel):
    idx: str
    label: Optional[str]
    instance_of: Optional[list[tuple[str, str]]]
    description: Optional[list[str]]
    image: Optional[list[str]]


class PipelineResponce(BaseModel):
    answers: list[str]


class EntityNeighboursResponce(BaseModel):
    entity: str
    property: str


class QuestionEntitiesResponce(BaseModel):
    entity: str
    neighbours: list[EntityNeighboursResponce]


class ACTPipelineResponce(PipelineResponce):
    instance_of_score: list[float]
    forward_one_hop_neighbours_score: list[float]
    answers_candidates_score: list[float]
    property_question_intersection_score: list[float]
    answers_candidates: list[str]
    answer_instance_of: list[str]
    answer_instance_of_count: dict[str, int]
    question_entities: list[QuestionEntitiesResponce]


class ACTPipelineResponceWithDescriptionScore(ACTPipelineResponce):
    entity_description_similarity_score: list[float]


class M3MPipelineResponce(PipelineResponce):
    scores: list[float]
    uncertenity: float
    triples: list[list[str]]


class WikidataSSPRequest(BaseModel):
    question_entities_idx: list[str]
    answer_idx: str

