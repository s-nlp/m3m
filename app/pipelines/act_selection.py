from functools import lru_cache

from fastapi import APIRouter
from joblib import Parallel, delayed
from pywikidata import Entity

from app.kgqa.act_selection import QuestionToRankInstanceOf
from app.kgqa.entity_linking import EntitiesSelection
from app.kgqa.mgenre import build_mgenre_pipeline
from app.kgqa.ner import NerToSentenceInsertion
from app.kgqa.utils import label_to_entity_idx
from app.models.base import Entity as EntityResponce
from app.models.base import Question as QuestionRequest
from app.models.base import ACTPipelineResponce
from app.pipelines.seq2seq import seq2seq

ner = NerToSentenceInsertion("/data/ner/")
mgenre = build_mgenre_pipeline()
entity_selection = EntitiesSelection(ner.model)


router = APIRouter(
    prefix="/pipeline/act_selection",
    tags=["pipeline", "act_selection"],
)


@router.post("/ner")
def ner_to_sentence_insertation(question: str):
    question_wit_ner, question_entities = ner.entity_labeling(question, True)
    return {
        "question": question_wit_ner,
        "entities": question_entities,
    }


@router.post("/mgenre")
def mgenre_linking(question: str):
    return mgenre(question)


@router.post("/entity_selection")
def entity_selection_mgenre_postprocess(
    question_entties: list[str],
    mgenre_predicted_entities: list[str],
):
    return entity_selection(question_entties, mgenre_predicted_entities)


@router.post("/seq2seq")
def raw_seq2seq(question: str) -> list[str]:
    return seq2seq(question)


@lru_cache(maxsize=1024)
@router.post("/")
def pipeline(question: QuestionRequest) -> ACTPipelineResponce:
    question_wit_ner, all_question_entities = ner.entity_labeling(question.text, True)
    mgenre_predicted_entities = mgenre(question_wit_ner)
    question_entities = entity_selection(
        all_question_entities, mgenre_predicted_entities
    )
    question_entities = [label_to_entity_idx(label) for label in question_entities]
    question_entities = [idx for idx in question_entities if idx is not None]

    seq2seq_results = seq2seq(question.text)
    answers_candidates = Parallel(n_jobs=-2)(
        delayed(label_to_entity_idx)(label) for label in seq2seq_results
    )
    answers_candidates = [e for e in answers_candidates if e is not None]

    question_to_rank = QuestionToRankInstanceOf(
        question=question.text,
        question_entities=question_entities,
        answers_candidates=answers_candidates,
        only_forward_one_hop=True,
    )
    answers = question_to_rank.final_answers()[:100]

    return ACTPipelineResponce(
        answers={a[1].idx for a in answers},
        answers_candidates=[e.idx for e in question_to_rank.answers_candidates],
        answer_instance_of=[e.idx for e in question_to_rank.answer_instance_of],
        answer_instance_of_count={e.idx:int(count) for e, count in question_to_rank.answer_instance_of_count},
        question_entities=[e.idx for e in question_to_rank.question_entities],
    )
