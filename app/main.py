from typing import Optional
from functools import lru_cache

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, Response
from fastapi.templating import Jinja2Templates
# from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware

from pywikidata import Entity

from .kgqa.seq2seq import build_seq2seq_pipeline
from .config import seq2seq as seq2seq_config

app = FastAPI()

templates = Jinja2Templates(directory="app/templates")

seq2seq_pipeline = build_seq2seq_pipeline(seq2seq_config["model"]["path"])

def _label_to_entity(label):
    try:
        return Entity.from_label(label)[0]
    except:
        pass

@lru_cache(maxsize=1024)
def seq2seq_question_to_answer_entities(question):
    # seq2seq_results = ['Donald Trump', 'Baiden', 'BBdn']
    seq2seq_results = seq2seq_pipeline(question)
    answers = [_label_to_entity(label) for label in seq2seq_results]
    answers = [a for a in answers if a is not None]
    return answers


@app.get("/", response_class=HTMLResponse)
async def index(request: Request, question: str = ''):
    if question != '':
        seq2seq_answers = seq2seq_question_to_answer_entities(question)
        final_answers = seq2seq_answers
        # print("final_answer description: ", final_answer.description)

        return templates.TemplateResponse(
            "index_search.html",
            {
                "request": request,
                "question": question,
                "final_answers": final_answers,
                "seq2seq_answers": seq2seq_answers,
            }
        )
    else:
        return templates.TemplateResponse(
            'index.html',
            {"request": request,}
        )

@lru_cache(maxsize=1024)
@app.post(f"/seq2seq/{seq2seq_config['model']['route_postfix']}")
def seq2seq(question: str):
    seq2seq_results = seq2seq_pipeline(question)
    # seq2seq_results = ['Trump', 'Baiden', 'BBdn']
    return seq2seq_results


@app.get(f"/wikidata/label_to_entity")
def label_to_entity(label: str):
    try:
        return Entity.from_label(label)[0].idx
    except:
        pass



