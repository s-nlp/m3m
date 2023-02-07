from functools import lru_cache
from typing import Dict, List

import torch
from fastapi import APIRouter
from joblib import Parallel, delayed
from pywikidata import Entity

from app.config import seq2seq as seq2seq_config
from app.kgqa.seq2seq import build_seq2seq_pipeline
from app.models.base import Entity as EntityResponce
from app.models.base import PipelineResponce
from app.models.base import Question as QuestionRequest

router = APIRouter(
    prefix="/pipeline/seq2seq",
    tags=["pipeline", "seq2seq"],
)

seq2seq = build_seq2seq_pipeline(
    seq2seq_config["model"]["path"],
    torch.device(seq2seq_config["device"]),
)


@lru_cache(maxsize=1024)
@router.post("/")
def seq2seq_pipeline(question: QuestionRequest) -> PipelineResponce:
    seq2seq_results = seq2seq(question.text)

    def _label_to_entity(label):
        try:
            return [e.idx for e in Entity.from_label(label)]
        except:
            pass

    corr_entities = Parallel(n_jobs=-2)(
        delayed(_label_to_entity)(label) for label in seq2seq_results
    )
    corr_entities = [e for e in corr_entities if e is not None]

    results = []
    for _, entities in zip(seq2seq_results, corr_entities):
        idx = list(sorted(entities, key=lambda idx: int(idx[1:])))[0]
        results.append(idx)

    return PipelineResponce(answers=results)
