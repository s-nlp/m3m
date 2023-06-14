from functools import lru_cache

import numpy as np
from fastapi import APIRouter
from joblib import Parallel, delayed
from pywikidata import Entity

from app.config import m3m as m3m_config
from app.kgqa.m3m import M3MQA, EncoderBERT
from app.models.base import M3MPipelineResponce
from app.models.base import Question as QuestionRequest


router = APIRouter(
    prefix="/pipeline/m3m",
    tags=["pipeline", "m3m"],
)

m3m = M3MQA(**m3m_config)


@lru_cache(maxsize=1024)
@router.post("/")
def m3m_pipeline(question: QuestionRequest) -> M3MPipelineResponce:
    triples, predicts_filtered, scores_filtered, ue = m3m(question.text)
    final_answers_idxs = []
    predicts_filtered_set = []
    for idx, pred in enumerate(predicts_filtered):
        if pred not in predicts_filtered_set:
            predicts_filtered_set.append(pred)
            final_answers_idxs.append(idx)

    return M3MPipelineResponce(
        answers=np.array(predicts_filtered)[final_answers_idxs][:50].tolist(),
        scores=np.array(scores_filtered)[final_answers_idxs][:50].tolist(),
        uncertenity=float(ue),
        triples=np.array(triples)[final_answers_idxs][:50].tolist(),
    )
