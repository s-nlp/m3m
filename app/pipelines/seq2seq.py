from functools import lru_cache
from typing import Dict, List

import torch
from fastapi import APIRouter
from joblib import Parallel, delayed

from app.config import seq2seq as seq2seq_config
from app.kgqa.entity_linking import entity_linker
from app.kgqa.seq2seq import build_seq2seq_pipeline
from app.kgqa.utils.utils import label_to_entity_idx
from app.models.base import Seq2SeqPipelineResponce
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
def seq2seq_pipeline(question: QuestionRequest) -> Seq2SeqPipelineResponce:
    question_entities = entity_linker.extract_entities_from_question(question.text)
    seq2seq_results = seq2seq(question.text)

    corr_entities = Parallel(n_jobs=-2)(
        delayed(label_to_entity_idx)(label) for label in seq2seq_results
    )
    corr_entities = [e for e in corr_entities if e is not None]
    return Seq2SeqPipelineResponce(answers=corr_entities[:60], question_entities=question_entities)
