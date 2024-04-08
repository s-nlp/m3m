import logging
import random
from typing import List

import numpy as np
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from internal.run_gap import GAPEvaluator
from utils import convert_to_webnlg_format

app = FastAPI()

origins = [
    "http://localhost:8085",
    "http://localhost",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

log_filename = "eval_log.txt"
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO,
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class WikidataSSPRequestEntity(BaseModel):
    id: str
    label: str
    type: str


class WikidataSSPRequestTriplet(BaseModel):
    source_entity: WikidataSSPRequestEntity
    relation: str
    target_entity: WikidataSSPRequestEntity


class WikidataSSPRequest(BaseModel):
    triplets: List[WikidataSSPRequestTriplet]


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


g2tModel = GAPEvaluator(
    args=AttrDict({
        "dataset": "webnlg",
        "output_dir": "internal/out",
        "entity_entity": True,
        "entity_relation": True,
        "tokenizer_path": "facebook/bart-base",
        "type_encoding": True,
        "max_node_length": 60,
        "predict_batch_size": 16,
        "max_input_length": 256,
        "max_output_length": 512,
        "append_another_bos": True,
        "num_beams": 5,
        "length_penalty": 5,
        # defaults
        "model_name": "bart",
        "do_lowercase": False,
        "remove_bos": False,
        "max_edge_length": 60,
        "clean_up_spaces": False,
        "prefix": "",
        "num_workers": 8,
        "relation_relation": False,
        "relation_entity": False,
        "debug": False,
        "n_gpu": torch.cuda.device_count(),
    }),
    logger=logger
)


@app.post("/graph/description")
async def ssp_subgraph_description(request: WikidataSSPRequest) -> str:
    webnlg_input = [convert_to_webnlg_format(request.triplets)]
    graph_description = g2tModel.run(webnlg_input)
    return graph_description

