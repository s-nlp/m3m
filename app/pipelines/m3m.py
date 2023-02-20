from functools import lru_cache

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
    predicts_filtered, scores_filtered, ue = m3m(question.text)

    return M3MPipelineResponce(
        answers=predicts_filtered[:60],
        scores=scores_filtered[:60],
        uncertenity=ue,
    )
