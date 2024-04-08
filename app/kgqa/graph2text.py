from functools import lru_cache
from typing import List, Tuple

import requests
from pywikidata import Entity
from transformers import pipeline
from app.config import g2t as g2t_config
from app.models.base import G2TModels

g2t_model = pipeline(
    task="text2text-generation",
    model=g2t_config["model"]["path"],
    device=g2t_config["device"],
    max_new_tokens=100
)


class Graph2Text:

    @staticmethod
    def __prepare_text(triplets: List[Tuple[str, str, str]], answer_id: str, question_ids: Tuple[str], model: G2TModels):
        graph = None
        triplets = list(map(lambda x: (Entity(x[0]), Entity(x[1]), Entity(x[2])), triplets))
        if model == G2TModels.T5_MODEL_NAME:
            graph = "[graph]"
            for triplet in triplets:
                graph += f"[head] {triplet[0].label} [relation] {triplet[1].label} [tail] {triplet[2].label} "
            graph += "[text]</s>"
        elif model == G2TModels.GAP_MODEL_NAME:
            graph = {
                "triplets": list(map(lambda x: Graph2Text.format_triplet_for_gap(x, answer_id, question_ids), triplets))
            }
        return graph

    @lru_cache(maxsize=1024)
    def __call__(self, input_graph, answer_id: str, question_ids: Tuple[str], model: G2TModels):
        description = ""
        graph = self.__prepare_text(input_graph, answer_id, question_ids, model)
        if model == G2TModels.T5_MODEL_NAME:
            result = g2t_model(graph)
            description = result[0]['generated_text']
        elif model == G2TModels.GAP_MODEL_NAME:
            print(f"http://localhost:{g2t_config['gap_port']}/graph/description")
            result = requests.post(f"http://{g2t_config['gap_host']}:{g2t_config['gap_port']}/graph/description", json=graph)
            print(result)
            description = result.text

        processed_description = description.replace(" ,", ",").replace(" .", ".").replace(" '", "'")
        return processed_description

    @staticmethod
    def format_triplet_for_gap(triplet: Tuple[Entity], answer_id: str, question_ids: Tuple[str]):
        source_entity, relation, target_entity = triplet
        source_entity = {
            "label": source_entity.label,
            "id": source_entity.idx,
            "type": "QUESTIONS_ENTITY" if source_entity.idx in question_ids
            else "ANSWER_CANDIDATE_ENTITY" if source_entity.idx == answer_id else "INTERNAL"
        }
        target_entity = {
            "label": target_entity.label,
            "id": target_entity.idx,
            "type": "QUESTIONS_ENTITY" if target_entity.idx in question_ids
            else "ANSWER_CANDIDATE_ENTITY" if target_entity.idx == answer_id else "INTERNAL"
        }
        return {
            "source_entity": source_entity,
            "target_entity": target_entity,
            "relation": relation.label
        }
