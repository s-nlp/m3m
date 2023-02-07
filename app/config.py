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
}


DEFAULT_CACHE_PATH = "/tmp/"
DEFAULT_LRU_CACHE_MAXSIZE = 8192
