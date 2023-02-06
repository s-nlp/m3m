from functools import lru_cache

from fastapi import APIRouter
from joblib import Parallel, delayed
from pywikidata import Entity

from app.kgqa.act_selection import QuestionToRankInstanceOf
from app.kgqa.entity_linking import EntitiesSelection
from app.kgqa.mgenre import build_mgenre_pipeline
from app.kgqa.ner import NerToSentenceInsertion
from app.models.base import Entity as EntityResponce
from app.models.base import PipelineResponce
from app.models.base import Question as QuestionRequest
from app.pipelines.seq2seq import seq2seq

ner = NerToSentenceInsertion("/data/ner/")
mgenre = build_mgenre_pipeline()
entity_selection = EntitiesSelection(ner.model)


router = APIRouter(
    prefix="/pipeline/act_selection",
    tags=["pipeline", "act_selection"],
)


class ACTPipelineResponce(PipelineResponce):
    type: list[str]


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
    question_wit_ner, all_question_entities = ner.entity_labeling(question, True)
    mgenre_predicted_entities = mgenre(question_wit_ner)
    question_entities = entity_selection(
        all_question_entities, mgenre_predicted_entities
    )
    seq2seq_results = seq2seq(question)

    def _label_to_entity(label):
        try:
            return [e.idx for e in Entity.from_label(label)]
        except:
            pass

    corr_entities = Parallel(n_jobs=-2)(
        delayed(_label_to_entity)(label) for label in seq2seq_results
    )
    answers_candidates = []
    for _, entities in zip(seq2seq_results, corr_entities):
        idx = list(sorted(entities, key=lambda idx: int(idx[1:])))[0]
        answers_candidates.append(idx)

    question_to_rank = QuestionToRankInstanceOf(
        question=question,
        question_entities=question_entities,
        answers_candidates=answers_candidates,
        only_forward_one_hop=True,
    )
    answers = question_to_rank.final_answers()

    return ACTPipelineResponce(
        answers={a[1].idx for a in answers},
        type=[e.idx for e in question_to_rank.answer_instance_of],
    )
