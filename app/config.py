import os

seq2seq = {
    "model": {
        "path": os.environ.get(
            "SEQ2SEQ_MODEL_PATH",
            "msalnikov/kgqa_sqwd-tunned_t5-large-ssm-nq",
        ),
        "route_postfix": "sqwd_tunned/t5_large_ssm_nq",
    },
    "examples": [],
    "device": "cuda:0",
    # "device": "cpu",
}

m3m = {
    "encoder_ckpt_path": "/data/m3m/ckpts/encoder.pt",
    "projection_e_ckpt_path": "/data/m3m/ckpts/projection_E",
    "projection_q_ckpt_path": "/data/m3m/ckpts/projection_Q",
    "projection_p_ckpt_path": "/data/m3m/ckpts/projection_P",
    "embeddings_path_q": "/data/m3m/table-qa/new_data/entitie_embeddings_ru_tensor.pt",
    "embeddings_path_p": "/data/m3m/table-qa/new_data/entitie_P_embeddings_ru_tensor.pt",
    "id2ind_path": '/data/m3m/table-qa/new_data/id2ind.npy',
    "p2ind_path": '/data/m3m/table-qa/new_data/p2ind.npy',
    "wikidata_cach_path": '/data/m3m/table-qa/wikidata_correct_cache.npy',
    "max_presearch": 7,
    "max_len_q": 64,
    "device": "cuda",
    # "device": "cpu",
    "aliases_leveldb_path": "/data/wikidata/aliases_lvldb/"
}

g2t = {
    "model": {
        "path": os.environ.get(
            "G2T_MODEL_PATH",
            "s-nlp/g2t-t5-xl-webnlg",
        )
    },
    # "device": "cuda",
    "device": "cpu",
}

# m3m = {
#     "encoder_ckpt_path": "s-nlp/m3m_bert_encoder",
#     "projection_e_ckpt_path": "/home/salnikov/data/m3m/ckpts/projection_E",
#     "projection_q_ckpt_path": "/home/salnikov/data/m3m/ckpts/projection_Q",
#     "projection_p_ckpt_path": "/home/salnikov/data/m3m/ckpts/projection_P",
#     "embeddings_path_q": "/home/salnikov/data/m3m/table-qa/new_data/entitie_embeddings_ru.json",
#     "embeddings_path_p": "/home/salnikov/data/m3m/table-qa/new_data/entitie_P_embeddings_ru.json",
#     "max_presearch": 7,
#     "max_len_q": 64,
#     "device": "cpu",
#  }


DEFAULT_CACHE_PATH = "/tmp/"
DEFAULT_LRU_CACHE_MAXSIZE = 8192
