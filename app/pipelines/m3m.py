from functools import lru_cache

import numpy as np
from fastapi import APIRouter, HTTPException
from joblib import Parallel, delayed
from pywikidata import Entity

from app.config import m3m as m3m_config
from app.kgqa.m3m import M3MQA, EncoderBERT
from app.models.base import M3MPipelineResponce
from app.models.base import Question as QuestionRequest

import torch
import torch.nn as nn
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from nltk.corpus import stopwords
import spacy
import plyvel
import pickle
import itertools
from langdetect import detect

from .act_selection import ner, mgenre, entity_selection
from app.kgqa.utils.utils import get_wd_search_results


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


class M3MQAmatching(M3MQA):
    nlp = spacy.load("xx_ent_wiki_sm")
    stopwords = list(itertools.chain.from_iterable([
        stopwords.words(lang) for lang in stopwords.fileids()
    ]))

    @staticmethod
    @lru_cache(maxsize=100000)
    def _split_toks(
        label,
    ): 
        # print("LABEL: ", label)
        # doc = M3MQAmatching.nlp(str(label))
        return label.lower().split()
        # return [token.lemma_ for token in doc]

    def __init__(
        self,
        encoder_ckpt_path: str,
        projection_e_ckpt_path: str,
        projection_q_ckpt_path: str,
        projection_p_ckpt_path: str,
        embeddings_path_q: str,
        embeddings_path_p: str,
        id2ind_path: str,
        p2ind_path: str,
        wikidata_cach_path: str,
        max_presearch: int = 7,
        max_len_q: int = 64,
        device: str = 'cpu',
        aliases_leveldb_path: str = '/data/wikidata/aliases_lvldb/',
        *args,
        **kwards,
    ):
        super().__init__(
            encoder_ckpt_path,
            projection_e_ckpt_path,
            projection_q_ckpt_path,
            projection_p_ckpt_path,
            embeddings_path_q,
            embeddings_path_p,
            id2ind_path,
            p2ind_path,
            wikidata_cach_path,
            max_presearch,
            max_len_q,
            device,
            *args,
            **kwards,
        )

        self.aliases_db = plyvel.DB(aliases_leveldb_path)

    def __call__(self, question: str):
        nouns = self.nouns_extractor(question)
        nouns = list(set(nouns))

        ids_q = []
        for noun in nouns:
            ids_q += get_wd_search_results(noun, self.max_presearch)
        ids_q = list(set(ids_q))

        if len(ids_q) == 0:
            for token in question.split():
                ids_q += get_wd_search_results(token, self.max_presearch)
        ids_q = list(set(ids_q))

        second_hop_ids_QP = dict()

        for idd in ids_q:
            if idd in self.wikidata_cache:
                triplets = self.wikidata_cache[idd]
                triplets = [(s, p) for p,s in zip(triplets[0:-1:2], triplets[1::2])]
                second_hop_ids_QP[idd] = triplets

        first_hop_graph_E = []
        second_hop_graph_Q = []
        second_hop_ids_filtered_Q = []
        second_hop_graph_P = []
        second_hop_ids_filtered_P = []
        ids_filtered_E = []
        for key in second_hop_ids_QP.keys():
            for (idd_q, idd_p) in second_hop_ids_QP[key]:
                if idd_q in self.id2ind and idd_p in self.p2ind and key in self.id2ind:
                    ids_filtered_E.append(key)
                    first_hop_graph_E.append(self.embeddings_tensor_Q[self.id2ind[key]].cpu().numpy())
                    second_hop_ids_filtered_Q.append(idd_q)
                    second_hop_graph_Q.append(self.embeddings_tensor_Q[self.id2ind[idd_q]].cpu().numpy())
                    second_hop_ids_filtered_P.append(idd_p)
                    second_hop_graph_P.append(self.embeddings_tensor_P[self.p2ind[idd_p]].cpu().numpy())


        if len(ids_filtered_E) == 0:
            return None

        predicts, scores, triples = self.get_top_ids_second_hop(
            question,
            first_hop_graph_E,
            second_hop_graph_Q,
            second_hop_graph_P,
            second_hop_ids_filtered_Q,
            second_hop_ids_filtered_P,
            ids_filtered_E,
            min(50,len(second_hop_graph_Q)),
        )
        
        UE = scores[0] - scores[1]
        return triples, predicts, scores.detach().cpu().numpy(), UE.item()


    def get_entity_labels(self, obj_id, lang='en'):
        obj_labels_aliases = self.aliases_db.get(obj_id.encode())
        if obj_labels_aliases is None:
            aliases = Entity(obj_id).attributes.get('aliases', {})

            labels_and_aliases = {}
            for vals in aliases.values():
                for val in vals:
                    if val['language'] not in labels_and_aliases:
                        labels_and_aliases[val['language']] = set()

                    labels_and_aliases[val['language']].add(val['value'])

            obj_labels_aliases = pickle.dumps(labels_and_aliases, pickle.HIGHEST_PROTOCOL)
            self.aliases_db.put(obj_id.encode(), obj_labels_aliases)
        
        if pickle.loads(obj_labels_aliases) is None:
            return set([])
        return pickle.loads(obj_labels_aliases).get(lang, set([]))
    
    def extract_entities(self, text):
        question_wit_ner, all_question_entities = ner.entity_labeling(text, True)
        print('NER RESULTS:>>', question_wit_ner, all_question_entities)
        mgenre_predicted_entities = mgenre(question_wit_ner)
        question_entities = entity_selection(
           all_question_entities, mgenre_predicted_entities
        )
        return question_entities

    def get_top_ids_second_hop(
        self,
        text,
        first_hop_graph_E,
        second_hop_graph_Q,
        second_hop_graph_P,
        second_hop_ids_filtered_Q,
        second_hop_ids_filtered_P,
        ids_filtered_E,
        topk,
    ):
        self.projection_E.eval()
        self.projection_P.eval()
        self.projection_Q.eval()
        
        with torch.no_grad():
            X = torch.tensor([self.tokenizer.encode(text, max_length=self.max_len_q, add_special_tokens=True,pad_to_max_length=True)]).to(self.device)[0].to(self.device)
            y_pred_e = self.projection_E(self.encoder(X[None,:].to(self.device)))
            y_pred_q = self.projection_Q(self.encoder(X[None,:].to(self.device)))
            y_pred_p = self.projection_P(self.encoder(X[None,:].to(self.device)))

            embeddings_tensor_E = torch.tensor(first_hop_graph_E, dtype=torch.float)
            embeddings_tensor_Q = torch.tensor(second_hop_graph_Q, dtype=torch.float)
            embeddings_tensor_P = torch.tensor(second_hop_graph_P, dtype=torch.float)

            cosines_descr_E = torch.cosine_similarity(embeddings_tensor_E.cpu(),y_pred_e.cpu())
            cosines_descr_E = nn.Softmax()(cosines_descr_E)

            cosines_descr_Q = torch.cosine_similarity(embeddings_tensor_Q.cpu(),y_pred_q.cpu())
            cosines_descr_Q = nn.Softmax()(cosines_descr_Q)

            cosines_descr_P = torch.cosine_similarity(embeddings_tensor_P.cpu(),y_pred_p.cpu())
            cosines_descr_P = nn.Softmax()(cosines_descr_P)

            # Additional score: Is tokens of question entity label in question or not
            matching_object_to_question_score = []
            # question_toks = M3MQAmatching._split_toks(text.replace('?', ''))
            question_entities = self.extract_entities(text)
            lang = detect(text)
            for obj_id in ids_filtered_E:
                obj_labels = self.get_entity_labels(obj_id, lang)
                obj_labels.add(Entity(obj_id).label)

                score = 0.0
                for obj_label in obj_labels:
                    obj_label_toks = set([token for token in M3MQAmatching._split_toks(obj_label) if token != ''])

                    for e in question_entities:
                        question_toks = [token for token in M3MQAmatching._split_toks(e) if token != '']
                        current_score = float(
                            len(obj_label_toks.intersection(question_toks)) >= max(1, round(len(obj_label_toks) / 2)) and
                            len(question_toks) > 0 and 
                            len(obj_label_toks) > 0
                        )
                        score = max(score, current_score)

                print("=> ", question_entities, text, lang, obj_id, obj_labels, score)
                matching_object_to_question_score.append(score)

            matching_object_to_question_score = torch.tensor(matching_object_to_question_score)
            # Additional score END

            cosines_aggr = cosines_descr_P + cosines_descr_Q + cosines_descr_E + matching_object_to_question_score
            inds = torch.topk(cosines_aggr,topk,sorted=True).indices.cpu().numpy()
            
            P = np.array(second_hop_ids_filtered_P)[inds]
            E = np.array(ids_filtered_E)[inds]
            Q = np.array(second_hop_ids_filtered_Q)[inds]
            final_triples = []
            for p, e, q in zip(P,E,Q):
                final_triples.append((e,p,q))
        return Q, cosines_aggr[inds], np.array(final_triples)

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
