import json
from multiprocessing import Manager, Process

import numpy as np
import spacy
import torch
import torch.nn as nn
from natasha import (Doc, MorphVocab, NamesExtractor, NewsEmbedding,
                     NewsMorphTagger, NewsNERTagger, NewsSyntaxParser,
                     Segmenter)
from natasha.doc import DocSpan
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from transformers import BertModel, BertTokenizer
from wikidata.client import Client as WDClient

from .utils.utils import get_wd_search_results


class EncoderBERT(nn.Module):
    def __init__(self):
        super(EncoderBERT,self).__init__()
        self.encoder =  BertModel.from_pretrained("bert-base-multilingual-cased")

    def forward(self, questions):
        q_ids = torch.tensor(questions)
        last_hidden_states = self.encoder(q_ids)[0]
        q_emb = last_hidden_states.mean(1)
        return q_emb


class KostilPhraseNormalization():
    def __init__(self):
        self.segmenter = Segmenter()
        self.morph_vocab = MorphVocab()

        self.emb = NewsEmbedding()
        self.morph_tagger = NewsMorphTagger(self.emb)
        self.syntax_parser = NewsSyntaxParser(self.emb)
        self.doc = None
        
    def phrase_preprocess(self, phrase):
        self.doc = Doc(phrase)
        self.doc.segment(self.segmenter)
        self.doc.tag_morph(self.morph_tagger)
        
    def get_tokens(self, phrase, tokens):
        local_tokens = phrase.split()
        result_tokens = []
        for token in tokens:
            if token.text in local_tokens:
                result_tokens.append(token)
        return result_tokens
    
    def normalize(self, phrase):
        self.phrase_preprocess(phrase)
        
        tokens = self.get_tokens(phrase, self.doc.tokens)
        span = DocSpan('0', '2', type='LOC', text=phrase, tokens=tokens)
        span.normalize(self.morph_vocab)
        return span.normal

    def __call__(self, phrase):
        return self.normalize(phrase)


class NounsExtractor():
    def __init__(self):
        self.stops = set(stopwords.words('english'))
        self.reg_tokenizer = RegexpTokenizer(r'\w+')
        self.nlp = spacy.load("ru_core_news_sm") 
        
        self.segmenter = Segmenter()
        self.morph_vocab = MorphVocab()

        emb = NewsEmbedding()
        self.morph_tagger = NewsMorphTagger(emb)
        self.syntax_parser = NewsSyntaxParser(emb)
        self.ner_tagger = NewsNERTagger(emb)

        self.names_extractor = NamesExtractor(self.morph_vocab)

        self.normalizer = KostilPhraseNormalization()
    
    def __call__(self, text: str):
        text = text.replace('?', '')
        doc = Doc(text)
        doc.segment(self.segmenter)
        doc.tag_morph(self.morph_tagger)
        doc.tag_ner(self.ner_tagger)
        for span in doc.spans:
            span.normalize(self.morph_vocab)
            
        ents = [str(ent.normal) for ent in doc.spans]
        
        text1 = text.replace('\'', '').replace('\"', '')
        doc = Doc(text1)
        doc.segment(self.segmenter)
        doc.tag_morph(self.morph_tagger)
        doc.tag_ner(self.ner_tagger)
        for span in doc.spans:
            span.normalize(self.morph_vocab)
        
        ents += [str(ent.normal) for ent in doc.spans]
        
        if '"' in text:
            text3 = text[text.find('"')+1:]
            text3 = text3[0:text3.find('"')]
            doc = Doc(text3)
            doc.segment(self.segmenter)
            doc.tag_morph(self.morph_tagger)
            doc.tag_ner(self.ner_tagger)
            for span in doc.spans:
                span.normalize(self.morph_vocab)

            ents += [str(ent.normal) for ent in doc.spans]
            ents += [str(self.normalizer.normalize(text3))]
            ents += [str(text3)]
            
        if '«' in text:
            text4 = text[text.find('«')+1:]
            text4 = text4[0:text4.find('»')]
            doc = Doc(text4)
            doc.segment(self.segmenter)
            doc.tag_morph(self.morph_tagger)
            doc.tag_ner(self.ner_tagger)
            for span in doc.spans:
                span.normalize(self.morph_vocab)

            ents += [str(ent.normal) for ent in doc.spans]
            ents += [str(self.normalizer.normalize(text4))]
            ents += [str(text4)]
        

        nouns_set = None
        if len(ents) == 0:
            doc = self.nlp(text)
            ents = [token.lemma_ for token in doc if token.pos_ == "NOUN" or token.pos_ == "PROPN"]
            
            bigrams = [self.normalizer.normalize(" ".join(b)) for b in zip(text.split(" ")[:-1], text.split(" ")[1:])]
            ents += bigrams
            unigrams = self.reg_tokenizer.tokenize(text)
            unigrams = [word for word in unigrams if not word in self.stops]
            ents += unigrams
            nouns_set = set(ents)
            
        if nouns_set and "" in nouns_set:
            nouns_set.remove("")
        return list(nouns_set)


class M3MQA():
    def __init__(
        self,
        encoder_ckpt_path: str = "ckpts/encoder",
        projection_e_ckpt_path: str = "ckpts/projection_E",
        projection_q_ckpt_path: str = "ckpts/projection_Q",
        projection_p_ckpt_path: str = "ckpts/projection_P",
        embeddings_path_q: str = "../table-qa/new_data/entitie_embeddings_ru.json",
        embeddings_path_p: str = "../table-qa/new_data/entitie_P_embeddings_ru.json",
        max_presearch: int = 7,
        max_len_q: int = 64,
        device: str = 'cpu',
    ):
        self.max_presearch = max_presearch
        self.max_len_q = max_len_q
        self.device = device

        self.nouns_extractor = NounsExtractor()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

        # self.encoder = torch.load(encoder_ckpt_path, map_location=torch.device(device)).to(device)
        self.encoder = EncoderBERT()
        self.encoder.load_state_dict(torch.load(encoder_ckpt_path, map_location=torch.device(device)))
        self.encoder.to(device)
        self.projection_E = torch.load(projection_e_ckpt_path, map_location=torch.device(device)).to(device)
        self.projection_Q = torch.load(projection_q_ckpt_path, map_location=torch.device(device)).to(device)
        self.projection_P = torch.load(projection_p_ckpt_path, map_location=torch.device(device)).to(device)

        self.graph_embeddings_Q, self.embeddings_tensor_Q = self._init_graph_embeddings(embeddings_path_q)
        self.graph_embeddings_P, self.embeddings_tensor_P = self._init_graph_embeddings(embeddings_path_p)

    def __call__(self, question: str):
        nouns = self.nouns_extractor(question)
        nouns = list(set(nouns))

        ids_q = []
        for noun in nouns:
            ids_q += get_wd_search_results(noun, self.max_presearch)
        ids_q = list(set(ids_q))
        # second_hop_ids_QP = Manager().dict()
        # processes = []
        # for idd_q in ids_q:
        #     processes.append(Process(target=self.mp_get_second_hop_entities_by_idd, args=(idd_q, second_hop_ids_QP)))
        # for p in processes:
        #     p.start()
        # for p in processes:
        #     p.join()
        second_hop_ids_QP = dict()
        for idd_q in ids_q:
            second_hop_ids_QP[idd_q] = self.mp_get_second_hop_entities_by_idd(idd_q)


        first_hop_graph_E = []
        second_hop_graph_Q = []
        second_hop_ids_filtered_Q = []
        second_hop_graph_P = []
        second_hop_ids_filtered_P = []
        ids_filtered_E = []
        for key in second_hop_ids_QP.keys():
            for (idd_q, idd_p) in second_hop_ids_QP[key]:
                if idd_q in self.graph_embeddings_Q and idd_p in self.graph_embeddings_P and key in self.graph_embeddings_Q:
                    ids_filtered_E.append(key)
                    first_hop_graph_E.append(self.graph_embeddings_Q[key])
                    second_hop_ids_filtered_Q.append(idd_q)
                    second_hop_graph_Q.append(self.graph_embeddings_Q[idd_q])
                    second_hop_ids_filtered_P.append(idd_p)
                    second_hop_graph_P.append(self.graph_embeddings_P[idd_p])


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

    def mp_get_second_hop_entities_by_idd(self, idd):
        client = WDClient() 
        entity = client.get(idd, load = True)
        rs = entity.attributes["claims"].keys()
        triples = []
        for r in rs:
            items = entity.attributes["claims"][r]
            for item in items:
                item = item["mainsnak"]
                if "datavalue" in item and "value" in item["datavalue"] and type(item["datavalue"]["value"]) == dict: # sometimes it is empty
                    value = item["datavalue"]["value"]
                    if 'entity-type' in value.keys():
                        o = value["id"]
                        triples.append((o,r))
        # d[idd] = triples
        return triples

    def get_top_ids_second_hop(
        self,
        text,
        first_hop_graph_E,
        second_hop_graph_Q,
        second_hop_graph_P,
        second_hop_ids_filtered_Q,
        second_hop_ids_filtered_P,
        ids_filtered_E
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

            cosines_aggr = cosines_descr_P + cosines_descr_Q + cosines_descr_E
            inds = torch.topk(cosines_aggr,topk,sorted=True).indices.cpu().numpy()
            
            P = second_hop_ids_filtered_P[inds]
            E = ids_filtered_E[inds]
            Q = second_hop_ids_filtered_Q[inds]
            final_triples = []
            for p, e, q in zip(P,E,Q):
                final_triples.append(e,p,q)
        return np.array(final_triples), Q, cosines_aggr[inds]
        

    def _init_graph_embeddings(self, embeddings_path: str):
        with open(embeddings_path, 'r') as file_handler:
            graph_embeddings = json.load(file_handler)

        embeddings = graph_embeddings
        ids_list = list(graph_embeddings.keys())
        embeddings = [embeddings[idx] for idx in ids_list]
        embeddings_tensor = torch.tensor(embeddings, dtype=torch.float)
        return graph_embeddings, embeddings_tensor
    
    def _init_encoder(self):
        # self.encoder = EncoderBERT()
        pass

    def _init_projectors(self):
        # Trainable projection module
        # self.projection_E = nn.Sequential(
        #     nn.Linear(768,512),
        #     nn.ELU(),
        #     nn.Linear(512,512),
        #     nn.ELU(),
        #     nn.Linear(512,200),
        # )

        # self.projection_Q = nn.Sequential(
        #     nn.Linear(768,512),
        #     nn.ELU(),
        #     nn.Linear(512,512),
        #     nn.ELU(),
        #     nn.Linear(512,200),
        # )

        # self.projection_P = nn.Sequential(
        #     nn.Linear(768,512),
        #     nn.ELU(),
        #     nn.Linear(512,512),
        #     nn.ELU(),
        #     nn.Linear(512,200),
        # )
        pass
