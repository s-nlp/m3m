FROM python:3.10 as requirements-stage

WORKDIR /tmp
RUN pip install poetry
COPY ./pyproject.toml ./poetry.lock* /tmp/
RUN poetry export -f requirements.txt --output requirements.txt --without-hashes

FROM nvidia/cuda:11.7.1-devel-ubuntu22.04

RUN apt-get update && apt-get install software-properties-common --yes && apt-get update
RUN add-apt-repository --yes ppa:deadsnakes/ppa && apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install --yes --quiet --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3.10-distutils \
    python3.10-lib2to3 \
    python3.10-gdbm \
    python3.10-tk \
    pip
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 999 \
    && update-alternatives --config python3 && ln -s /usr/bin/python3 /usr/bin/python
RUN apt-get update && apt-get install libsnappy-dev libleveldb-dev libleveldb-dev --yes

RUN pip install --upgrade pip

WORKDIR /code
COPY --from=requirements-stage /tmp/requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
RUN apt-get update && apt-get install -y graphviz
RUN python3 -c "import nltk; nltk.download('stopwords')"
RUN python3 -m spacy download ru_core_news_sm && \
    python3 -m spacy download en_core_web_sm && \
    python3 -m spacy download xx_ent_wiki_sm && \
    python3 -m spacy download en_core_web_trf

COPY ./app /code/app
COPY ./data /data/

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--forwarded-allow-ips='*'", "--proxy-headers"]