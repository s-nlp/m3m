from pydantic import Field
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, Response
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from pywikidata import Entity

from app.kgqa.m3m import M3MQA, EncoderBERT
from app.pipelines import act_selection, seq2seq, m3m
from app.models.base import Entity as EntityResponce

app = FastAPI()

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


# @app.get("/wikidata/entities/{idx}/label")
# async def entity_label(idx: str) -> str:
#     entity = Entity(idx)
#     return entity.label
