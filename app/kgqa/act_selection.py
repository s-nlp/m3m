# pylint: disable=line-too-long,missing-class-docstring,invalid-name,redefined-builtin
from collections import defaultdict
from typing import List, Union, Optional, Tuple
from functools import lru_cache
from joblib import Memory

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from pywikidata import Entity
from app.config import DEFAULT_CACHE_PATH, DEFAULT_LRU_CACHE_MAXSIZE

memory = Memory(DEFAULT_CACHE_PATH, verbose=0)


INSTANCE_OF_IDX_BLACKLIST = [
    "Q4167410",  #   Wikimedia disambiguation page
    "Q14204246",  #  Wikimedia project page
    "Q35252665",  #  Wikimedia non-main namespace
    "Q11266439",  #  Wikimedia template
    "Q58494026",  #  Wikimedia page
    "Q17379835",  #  Wikimedia page outside the main knowledge tree
    "Q37152856",  #  Wikimedia page relating two or more distinct concepts
    "Q100510764",  # Wikibooks book
    "Q104696061",  # Wikibook page
    "Q114612576",  # Wikiversity course
    "Q115491908",  # Wikimedia subpage
    "Q115668764",  # Wiktionary rhymes page
    "Q15407973",  #  Wikimedia disambiguation category
    "Q22808320",  #  Wikimedia human name disambiguation page
    "Q61996773",  #  municipality name disambiguation page
    "Q66480449",  #  Wikimedia surname disambiguation page
    "Q15407973",  #  Wikimedia disambiguation category
]


class _QuestionToRankBase:
    def __init__(
        self,
        question: str,
        question_entities: Union[List[str], List[Entity]],
        answers_candidates: Union[List[str], List[Entity]],
        target_entity: Optional[Entity] = None,
    ):
        self.question = question
        self.question_entities = [
            e if isinstance(e, Entity) else Entity(e) for e in question_entities
        ]
        self.answers_candidates = [
            e if isinstance(e, Entity) else Entity(e) for e in answers_candidates
        ]

        if target_entity is not None:
            self.target = target_entity
        else:
            self.target = None

        self._answer_instance_of_count = None
        self._answer_instance_of = None
        self._final_answers = None

    @property
    def answer_instance_of_count(self):
        if self._answer_instance_of is None or self._answer_instance_of_count is None:
            self._calculate_answer_instance_of()

        return self._answer_instance_of_count

    @property
    def answer_instance_of(self):
        if self._answer_instance_of is None or self._answer_instance_of_count is None:
            self._calculate_answer_instance_of()

        return self._answer_instance_of

    def final_answers(self) -> List[Entity]:
        raise NotImplementedError()

    def _calculate_answer_instance_of(self):
        if self._answer_instance_of is None or self._answer_instance_of_count is None:
            self._answer_instance_of_count = defaultdict(float)
            for answer_entity in self.answers_candidates:
                for instance_of_entity in answer_entity.instance_of:
                    self._answer_instance_of_count[instance_of_entity] += 1

            self._answer_instance_of_count = sorted(
                self._answer_instance_of_count.items(), key=lambda v: -v[1]
            )
            self._answer_instance_of_count = [
                (key, val)
                for key, val in self._answer_instance_of_count
                if key.idx not in INSTANCE_OF_IDX_BLACKLIST
            ]

            self._answer_instance_of = self._select_answer_instance_of(
                self._answer_instance_of_count
            )

    def _select_answer_instance_of(
        self, answer_instance_of_count: List[Tuple[Entity, int]]
    ) -> List[Entity]:
        raise NotImplementedError()

    def __repr__(self) -> str:
        if self.target is not None:
            return f"<QuestionRank: {self.question} | {self.target}>"
        else:
            return f"<QuestionRank: {self.question}>"


class QuestionToRankInstanceOf(_QuestionToRankBase):
    stopwords = stopwords.words("english")
    stemmer = PorterStemmer()
    sent_transformer = SentenceTransformer("bert-base-nli-mean-tokens")

    def __init__(
        self,
        question: str,
        question_entities: Union[List[str], List[Entity]],
        answers_candidates: Union[List[str], List[Entity]],
        target_entity: Optional[Entity] = None,
        only_forward_one_hop: bool = False,
    ):
        super().__init__(question, question_entities, answers_candidates, target_entity)

        self.only_forward_one_hop = only_forward_one_hop

    @staticmethod
    @lru_cache(maxsize=DEFAULT_LRU_CACHE_MAXSIZE)
    def _split_toks(
        label,
    ):
        return [
            QuestionToRankInstanceOf.stemmer.stem(tok.lower())
            for tok in label.split()
            if QuestionToRankInstanceOf.stemmer.stem(tok.lower())
            not in QuestionToRankInstanceOf.stopwords
        ]

    @staticmethod
    @lru_cache(maxsize=DEFAULT_LRU_CACHE_MAXSIZE)
    @memory.cache
    def _calculate_sent_scores(tuple_of_strings):
        return QuestionToRankInstanceOf.sent_transformer.encode(tuple_of_strings)

    def _select_answer_instance_of(
        self, answer_instance_of_count: List[Tuple[Entity, int]]
    ) -> List[Entity]:
        initial_number = 3
        th = 0.6

        if len(answer_instance_of_count) <= initial_number:
            return [e for e, _ in answer_instance_of_count]

        sentence_embeddings = QuestionToRankInstanceOf._calculate_sent_scores(
            tuple([str(e.label) for e, _ in answer_instance_of_count])
        )
        scores_collection = cosine_similarity(
            sentence_embeddings[:initial_number], sentence_embeddings[initial_number:]
        )

        selected_entities = [e for e, _ in answer_instance_of_count[:initial_number]]
        for scores in scores_collection:
            for e, _ in np.array(answer_instance_of_count[initial_number:])[
                scores > th
            ]:
                if e not in selected_entities:
                    selected_entities.append(e)

        return selected_entities

    @staticmethod
    @lru_cache(maxsize=DEFAULT_LRU_CACHE_MAXSIZE)
    @memory.cache
    def _calculate_property_question_intersection_score(property, question):
        sentence_embeddings = QuestionToRankInstanceOf.sent_transformer.encode(
            [question, property.label]
        )
        score = cosine_similarity([sentence_embeddings[0]], [sentence_embeddings[1]])[
            0
        ][0]
        return score

    def final_answers(self) -> List[Entity]:
        if self._final_answers is None:
            selected_set = []
            for q_entity in self.question_entities:
                if self.only_forward_one_hop:
                    neighbours = q_entity.forward_one_hop_neighbours
                else:
                    neighbours = (
                        q_entity.forward_one_hop_neighbours
                        + q_entity.backward_one_hop_neighbours
                    )

                for property, entity in neighbours:
                    property_question_intersection_score = QuestionToRankInstanceOf._calculate_property_question_intersection_score(
                        property,
                        self.question,
                    )

                    instance_of_score = len(
                        set(entity.instance_of).intersection(list(self.answer_instance_of))
                    )
                    if instance_of_score > 0:
                        instance_of_score = (
                            len(self.answer_instance_of) - instance_of_score
                        ) / len(self.answer_instance_of)
                    # ---
                    if (property, entity) in q_entity.forward_one_hop_neighbours:
                        forward_one_hop_neighbours_score = 1
                    else:
                        forward_one_hop_neighbours_score = 0

                    # ---
                    if entity in self.answers_candidates:
                        count = len(self.answers_candidates)
                        answers_candidates_score = (
                            count - self.answers_candidates.index(entity)
                        ) / count
                    else:
                        answers_candidates_score = 0

                    ###
                    selected_set.append(
                        (
                            property,
                            entity,
                            instance_of_score,
                            forward_one_hop_neighbours_score,
                            answers_candidates_score,
                            property_question_intersection_score,
                        )
                    )

            for entity in self.answers_candidates:
                count = len(self.answers_candidates)
                answers_candidates_score = (
                    count - self.answers_candidates.index(entity)
                ) / count

                instance_of_score = len(
                    set(entity.instance_of).intersection(list(self.answer_instance_of))
                )
                if instance_of_score > 0:
                    instance_of_score = (
                        len(self.answer_instance_of) - instance_of_score
                    ) / len(self.answer_instance_of)

                selected_set.append(
                    (None, entity, instance_of_score, 0, answers_candidates_score, 0)
                )

            selected_set = sorted(selected_set, key=lambda x: -sum(x[2:]))
            self._final_answers = selected_set

        return list(self._final_answers)


class QuestionToRankInstanceOfSimple(QuestionToRankInstanceOf):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def _select_answer_instance_of(
        self, answer_instance_of_count: List[Tuple[Entity, int]]
    ) -> List[Entity]:
        initial_number = 3

        if len(answer_instance_of_count) == 0:
            return []

        selected_entities = []
        prev_count = answer_instance_of_count[0][1]
        for entity, count in answer_instance_of_count[:initial_number]:
            if prev_count - count > prev_count // 2:
                break
            selected_entities.append(entity)

        return selected_entities

    def _calculate_answer_instance_of(self):
        if self._answer_instance_of is None or self._answer_instance_of_count is None:
            self._answer_instance_of_count = defaultdict(float)
            for answer_entity in self.answers_candidates:
                for instance_of_entity in answer_entity.instance_of:
                    self._answer_instance_of_count[instance_of_entity] += 1

            self._answer_instance_of_count = sorted(
                self._answer_instance_of_count.items(), key=lambda v: -v[1]
            )
            self._answer_instance_of_count = [
                (key, val)
                for key, val in self._answer_instance_of_count
                if key.idx not in INSTANCE_OF_IDX_BLACKLIST + ['Q22808320']
            ]

            self._answer_instance_of = self._select_answer_instance_of(
                self._answer_instance_of_count
            )


class QuestionToRankInstanceOfSimpleWithDescriptionMatching(QuestionToRankInstanceOfSimple):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    @lru_cache(maxsize=DEFAULT_LRU_CACHE_MAXSIZE)
    @memory.cache
    def _calculate_entity_description_similarity_score(entity: Entity, question: str):
        if entity.description is None or len(entity.description) == 0:
            return 0
        
        description = " ".join(entity.description)
        sentence_embeddings = QuestionToRankInstanceOf.sent_transformer.encode(
            [question, description]
        )
        score = cosine_similarity([sentence_embeddings[0]], [sentence_embeddings[1]])[
            0
        ][0]
        return score

    def final_answers(self) -> List[Entity]:
        if self._final_answers is None:
            selected_set = []
            for q_entity in self.question_entities:
                if self.only_forward_one_hop:
                    neighbours = q_entity.forward_one_hop_neighbours
                else:
                    neighbours = (
                        q_entity.forward_one_hop_neighbours
                        + q_entity.backward_one_hop_neighbours
                    )

                for property, entity in neighbours:
                    property_question_intersection_score = QuestionToRankInstanceOf._calculate_property_question_intersection_score(
                        property,
                        self.question,
                    )

                    instance_of_score = len(
                        set(entity.instance_of).intersection(list(self.answer_instance_of))
                    )
                    if instance_of_score > 0:
                        instance_of_score = (
                            len(self.answer_instance_of) - instance_of_score
                        ) / len(self.answer_instance_of)
                    # ---
                    if (property, entity) in q_entity.forward_one_hop_neighbours:
                        forward_one_hop_neighbours_score = 1
                    else:
                        forward_one_hop_neighbours_score = 0

                    # ---
                    if entity in self.answers_candidates:
                        count = len(self.answers_candidates)
                        answers_candidates_score = (
                            count - self.answers_candidates.index(entity)
                        ) / count
                    else:
                        answers_candidates_score = 0

                    # ---
                    entity_description_similarity_score = QuestionToRankInstanceOfSimpleWithDescriptionMatching._calculate_entity_description_similarity_score(
                        entity,
                        self.question,
                    )

                    ###
                    selected_set.append(
                        (
                            property,
                            entity,
                            instance_of_score * 3,
                            forward_one_hop_neighbours_score * 3,
                            answers_candidates_score,
                            property_question_intersection_score,
                            entity_description_similarity_score * 3,
                        )
                    )

            for entity in self.answers_candidates:
                count = len(self.answers_candidates)
                answers_candidates_score = (
                    count - self.answers_candidates.index(entity)
                ) / count

                instance_of_score = len(
                    set(entity.instance_of).intersection(list(self.answer_instance_of))
                )
                if instance_of_score > 0:
                    instance_of_score = (
                        len(self.answer_instance_of) - instance_of_score
                    ) / len(self.answer_instance_of)

                entity_description_similarity_score = QuestionToRankInstanceOfSimpleWithDescriptionMatching._calculate_entity_description_similarity_score(
                    entity,
                    self.question,
                )

                selected_set.append(
                    (None, entity, instance_of_score * 3, 0, answers_candidates_score, 0, entity_description_similarity_score * 3)
                )

            selected_set = sorted(selected_set, key=lambda x: -sum(x[2:]))
            self._final_answers = selected_set

        return list(self._final_answers)