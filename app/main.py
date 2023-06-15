from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import Field
from pywikidata import Entity

from app.kgqa.m3m import M3MQA, EncoderBERT
from app.kgqa.utils.graph_viz import SubgraphsRetriever, plot_graph_svg
from app.models.base import Entity as EntityResponce
from app.models.base import WikidataSSPRequest
from app.pipelines import seq2seq
from app.pipelines import act_selection
from app.pipelines import m3m
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
app.mount("/static", StaticFiles(directory="app/static"), name="static")

app.include_router(act_selection.router)
app.include_router(seq2seq.router)
app.include_router(m3m.router)


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


@app.post("/wikidata/entities/ssp/graph/svg")
async def ssp_subgraph_svg(request: WikidataSSPRequest) -> str:
    sr = SubgraphsRetriever()
    graph, _ = sr.get_subgraph(request.question_entities_idx, request.answer_idx)
    graph_svg = plot_graph_svg(graph)
    return graph_svg.pipe(format='svg').replace(b'\n', b'')

