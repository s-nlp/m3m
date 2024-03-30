from functools import lru_cache
from typing import List, Tuple

from transformers import pipeline
from app.config import g2t as g2t_config


g2t_model = pipeline(
    task="text2text-generation",
    model=g2t_config["model"]["path"],
    device=g2t_config["device"]
)


class Graph2Text:

    @staticmethod
    def __prepare_text(triplets: List[Tuple[str, str, str]]):
        graph = "[graph]"
        for triplet in triplets:
            graph += f"[head] {triplet[0]} [relation] {triplet[1]} [tail] {triplet[2]} "
        graph += "[text]</s>"
        return graph

    @lru_cache(maxsize=1024)
    def __call__(self, input_graph):
        graph = self.__prepare_text(input_graph)
        result = g2t_model(graph)
        return result[0]['generated_text']
