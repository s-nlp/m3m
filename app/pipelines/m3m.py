import itertools
import pickle
from functools import lru_cache

import numpy as np
import plyvel
import spacy
import torch
import torch.nn as nn
from fastapi import APIRouter, HTTPException
from joblib import Parallel, delayed
from langdetect import detect
from nltk.corpus import stopwords
from pywikidata import Entity
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from app.config import m3m as m3m_config
from app.kgqa.m3m import M3MQA, EncoderBERT, M3MQAmatching
from app.kgqa.utils.utils import get_wd_search_results
from app.models.base import M3MPipelineResponce
from app.models.base import Question as QuestionRequest

from .act_selection import entity_selection, mgenre, ner

router = APIRouter(
    prefix="/pipeline/m3m",
    tags=["pipeline", "m3m"],
)

m3m = M3MQA(**m3m_config)


@lru_cache(maxsize=1024)
@router.post("/")
def m3m_pipeline(question: QuestionRequest) -> M3MPipelineResponce:
    m3m_results = m3m_matching(question.text)
    if m3m_results is None:
        raise HTTPException(status_code=404, detail='No object found in Wikidata for provided question nounces')
    else:
        triples, predicts_filtered, scores_filtered, ue = m3m_results
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

m3m_matching = M3MQAmatching(**m3m_config)

@lru_cache(maxsize=1024)
@router.post("/m3m_subj_question_matching")
def m3m_object_label_to_question_matching_pipeline(question: QuestionRequest) -> M3MPipelineResponce:
    m3m_results = m3m_matching(question.text)
    if m3m_results is None:
        raise HTTPException(status_code=404, detail='No object found in Wikidata for provided question nounces')
    else:
        triples, predicts_filtered, scores_filtered, ue = m3m_results
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
