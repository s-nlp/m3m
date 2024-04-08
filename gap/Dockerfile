FROM nvidia/cuda:11.7.1-devel-ubuntu22.04

RUN apt-get update && apt-get install software-properties-common --yes && apt-get update
RUN add-apt-repository --yes ppa:deadsnakes/ppa && apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install --yes --quiet --no-install-recommends \
    python3.7 \
    python3.7-dev \
    python3.7-distutils \
    python3.7-lib2to3 \
    python3.7-gdbm \
    python3.7-tk \
    pip
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 999 \
    && update-alternatives --config python3 && ln -s /usr/bin/python3 /usr/bin/python
#RUN apt-get update && apt-get install libsnappy-dev libleveldb-dev libleveldb-dev --yes

RUN pip install --upgrade pip

WORKDIR /code
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt && pip install --no-cache-dir torch-scatter==2.0.9
COPY . /code

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8089", "--forwarded-allow-ips='*'", "--proxy-headers"]