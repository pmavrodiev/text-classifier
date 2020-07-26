FROM continuumio/miniconda:latest

RUN apt-get -y update && apt-get -y install --reinstall build-essential && apt-get -y install unzip htop


ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN mkdir -p /app
RUN mkdir -p /app/simple_logging

COPY /src/app /app
COPY /src/simple_logging /app/simple_logging

RUN /opt/conda/bin/conda env create -f /app/requirements.yml
ENV PATH /opt/conda/envs/app/bin:$PATH

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONPATH /app

RUN echo "source activate app" >~/.bashrc

WORKDIR /app
CMD tail -f /dev/null

