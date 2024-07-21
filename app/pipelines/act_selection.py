from functools import lru_cache

from fastapi import APIRouter
from joblib import Parallel, delayed

from app.kgqa.act_selection import QuestionToRankInstanceOf, QuestionToRankInstanceOfSimple, QuestionToRankInstanceOfSimpleWithDescriptionMatching
from app.kgqa.entity_linking import ner, mgenre, entity_selection, entity_linker
from app.kgqa.utils.utils import get_wd_search_results
from app.models.base import Question as QuestionRequest
from app.models.base import ACTPipelineResponce, ACTPipelineResponceWithDescriptionScore, QuestionEntitiesResponce, EntityNeighboursResponce
from app.pipelines.seq2seq import seq2seq


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


def _prepare_question_entities_and_answer_candidate_helper(question: QuestionRequest):
    question_entities = entity_linker.extract_entities_from_question(question.text)
    seq2seq_results = seq2seq(question.text)
    answers_candidates = Parallel(n_jobs=-2)(
        delayed(get_wd_search_results)(label, 1) for label in seq2seq_results
    )
    answers_candidates = [e[0] for e in answers_candidates if e is not None and len(e) > 0]

    return question_entities, answers_candidates


@lru_cache(maxsize=1024)
@router.post("/main/")
def pipeline(question: QuestionRequest) -> ACTPipelineResponce:
    question_entities, answers_candidates = _prepare_question_entities_and_answer_candidate_helper(question)

    question_to_rank = QuestionToRankInstanceOf(
        question=question.text,
        question_entities=question_entities,
        answers_candidates=answers_candidates,
        only_forward_one_hop=True,
    )
    answers_with_scores = question_to_rank.final_answers()[:100]

    return ACTPipelineResponce(
        answers=[a[1].idx for a in answers_with_scores],
        instance_of_score=[a[2] for a in answers_with_scores],
        forward_one_hop_neighbours_score=[a[3] for a in answers_with_scores],
        answers_candidates_score=[a[4] for a in answers_with_scores],
        property_question_intersection_score=[a[5] for a in answers_with_scores],
        answers_candidates=[e.idx for e in question_to_rank.answers_candidates],
        answer_instance_of=[e.idx for e in question_to_rank.answer_instance_of],
        answer_instance_of_count={e.idx:int(count) for e, count in question_to_rank.answer_instance_of_count},
        question_entities=[
            QuestionEntitiesResponce(
                entity=e.idx,
                neighbours=[
                    EntityNeighboursResponce(entity=entity_neighbour.idx, property=prop.idx)
                    for prop, entity_neighbour in e.forward_one_hop_neighbours
                ]
            )
            for e in question_to_rank.question_entities
        ]
    )

@lru_cache(maxsize=1024)
@router.post("/simple_type_selection/")
def pipeline(question: QuestionRequest) -> ACTPipelineResponce:
    question_entities, answers_candidates = _prepare_question_entities_and_answer_candidate_helper(question)

    question_to_rank = QuestionToRankInstanceOfSimple(
        question=question.text,
        question_entities=question_entities,
        answers_candidates=answers_candidates,
        only_forward_one_hop=True,
    )
    answers_with_scores = question_to_rank.final_answers()[:60]

    return ACTPipelineResponce(
        answers=[a[1].idx for a in answers_with_scores],
        instance_of_score=[a[2] for a in answers_with_scores],
        forward_one_hop_neighbours_score=[a[3] for a in answers_with_scores],
        answers_candidates_score=[a[4] for a in answers_with_scores],
        property_question_intersection_score=[a[5] for a in answers_with_scores],
        answers_candidates=[e.idx for e in question_to_rank.answers_candidates],
        answer_instance_of=[e.idx for e in question_to_rank.answer_instance_of],
        answer_instance_of_count={e.idx:int(count) for e, count in question_to_rank.answer_instance_of_count},
        question_entities=[
            QuestionEntitiesResponce(
                entity=e.idx,
                neighbours=[
                    EntityNeighboursResponce(entity=entity_neighbour.idx, property=prop.idx)
                    for prop, entity_neighbour in e.forward_one_hop_neighbours
                ]
            )
            for e in question_to_rank.question_entities
        ]
    )


@lru_cache(maxsize=1024)
@router.post("/simple_with_description_qustion_similarity_type_selection/")
def pipeline(question: QuestionRequest) -> ACTPipelineResponceWithDescriptionScore:
    question_entities, answers_candidates = _prepare_question_entities_and_answer_candidate_helper(question)

    question_to_rank = QuestionToRankInstanceOfSimpleWithDescriptionMatching(
        question=question.text,
        question_entities=question_entities,
        answers_candidates=answers_candidates,
        only_forward_one_hop=True,
    )
    answers_with_scores = question_to_rank.final_answers()[:60]

    return ACTPipelineResponceWithDescriptionScore(
        answers=[a[1].idx for a in answers_with_scores],
        instance_of_score=[a[2] for a in answers_with_scores],
        forward_one_hop_neighbours_score=[a[3] for a in answers_with_scores],
        answers_candidates_score=[a[4] for a in answers_with_scores],
        property_question_intersection_score=[a[5] for a in answers_with_scores],
        entity_description_similarity_score=[a[6] for a in answers_with_scores],
        answers_candidates=[e.idx for e in question_to_rank.answers_candidates],
        answer_instance_of=[e.idx for e in question_to_rank.answer_instance_of],
        answer_instance_of_count={e.idx:int(count) for e, count in question_to_rank.answer_instance_of_count},
        question_entities=[
            QuestionEntitiesResponce(
                entity=e.idx,
                neighbours=[
                    EntityNeighboursResponce(entity=entity_neighbour.idx, property=prop.idx)
                    for prop, entity_neighbour in e.forward_one_hop_neighbours
                ]
            )
            for e in question_to_rank.question_entities
        ]
    )