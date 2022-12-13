FROM nvidia/cuda:11.7.0-runtime-ubuntu22.04
LABEL maintainer="pazyx728@gmail.com" \
      version="v0.1"

ENV DEBIAN_FRONTEND=noninteractive
RUN \
    --mount=type=cache,target=/var/cache/apt \
    apt-get update
RUN \
    --mount=type=cache,target=/var/cache/apt \
    apt-get -y --no-install-recommends install python3-pip
RUN \
    --mount=type=cache,target=/var/cache/apt \
    apt-get -y --no-install-recommends install wget unzip git software-properties-common \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev patchelf swig
RUN ln -s /usr/bin/python3 /usr/bin/python

RUN pip install torch==1.13.0 torchvision torchaudio

ARG REQ_DIR=requirements/requirements-dev.txt
RUN echo REQ_DIR=$REQ_DIR
ADD $REQ_DIR requirements.txt

RUN pip install -r requirements.txt &&\
    rm -rf $HOME/.cache/pip

ADD ./ abcdrl

WORKDIR abcdrl
