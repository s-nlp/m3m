from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import Field
from pywikidata import Entity

from app.kgqa.graph2text import Graph2Text
from app.kgqa.m3m import M3MQA, EncoderBERT
from app.kgqa.utils.graph_viz import SubgraphsRetriever, plot_graph_svg
from app.kgqa.utils.utils import validate_or_search_entity_idx
from app.models.base import Entity as EntityResponce, WikidataG2TRequest
from app.models.base import WikidataSSPRequest
from app.pipelines import seq2seq
from app.pipelines import act_selection
# from app.pipelines import m3m
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://kgqa-nlp-zh.skoltech.ru",
    "https://kgqa-nlp-zh.skoltech.ru",
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

templates = Jinja2Templates(directory="app/templates")
# app.mount("/static", StaticFiles(directory="app/static"), name="static")

app.include_router(act_selection.router)
app.include_router(seq2seq.router)
# app.include_router(m3m.router)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request, question: str = "", pipeline: str = "seq2seq"):
    if question != "":
        return templates.TemplateResponse(
            "index_search.html",
            {
                "request": request,
                "question": question,
                "pipeline": pipeline,
            },
        )
    else:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
            },
        )


@app.get("/graph", response_class=HTMLResponse)
async def graph(request: Request):
    return templates.TemplateResponse(
        "graph_visualise.html",
        {
            "request": request
        },
    )


@app.get("/wikidata/entities/{idx}")
async def entity_details(idx: str) -> EntityResponce:
    entity = Entity(idx)
    return EntityResponce(
        idx=entity.idx,
        label=entity.label,
        description=entity.description,
        image=entity.image,
        instance_of=[(e.idx, e.label) for e in entity.instance_of],
    )


@app.get("/wikidata/entities/{idx}/label")
async def entity_label(idx: str) -> EntityResponce:
    entity = Entity(idx)
    return EntityResponce(
        idx=entity.idx,
        label=entity.label,
    )


@app.post("/wikidata/entities/ssp/graph/description")
async def ssp_subgraph_description(request: WikidataG2TRequest) -> str:
    sr = SubgraphsRetriever()
    g2t = Graph2Text()
    verified_question_entities_idx = [validate_or_search_entity_idx(entity) for entity in request.question_entities_idx]
    verified_question_entities_idx = list(filter(lambda x: x is not None, verified_question_entities_idx))
    verified_answer_idx = validate_or_search_entity_idx(request.answer_idx)
    if len(verified_question_entities_idx) == 0 or verified_answer_idx is None:
        raise HTTPException(status_code=400, detail="Bad input entities")

    graph, _ = sr.get_subgraph(verified_question_entities_idx, verified_answer_idx)
    triplets = [(x, weight, y) for (x, y, weight) in graph.edges.data('label')]
    triplets = tuple(triplets)  # For cache
    graph_description = g2t(triplets, request.answer_idx, tuple(request.question_entities_idx), request.model)
    return graph_description


@app.post("/wikidata/entities/ssp/graph/svg")
async def ssp_subgraph_svg(request: WikidataSSPRequest) -> str:
    sr = SubgraphsRetriever()
    verified_question_entities_idx = [validate_or_search_entity_idx(entity) for entity in request.question_entities_idx]
    verified_question_entities_idx = list(filter(lambda x: x is not None, verified_question_entities_idx))
    verified_answer_idx = validate_or_search_entity_idx(request.answer_idx)
    if len(verified_question_entities_idx) == 0 or verified_answer_idx is None:
        raise HTTPException(status_code=400, detail="Bad input entities")

    graph, _ = sr.get_subgraph(verified_question_entities_idx, verified_answer_idx)
    graph_svg = plot_graph_svg(graph)
    return graph_svg.pipe(format='svg').replace(b'\n', b'')

