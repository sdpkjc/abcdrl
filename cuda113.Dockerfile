FROM nvidia/cuda:11.3.1-runtime-ubuntu18.04
LABEL maintainer="pazyx728@gmail.com"

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get -y install \
        python3.9 python3-pip wget unzip \
        git \
        libgl1-mesa-dev \
        libgl1-mesa-glx \
        libglew-dev \
        libosmesa6-dev patchelf swig && \
    apt-get autoclean && rm -rf /var/lib/apt/lists/*

RUN pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
ARG REQ_DIR=requirements/requirements-dev.txt
ADD $REQ_DIR requirements.txt
RUN pip install -r requirements.txt && \
    rm -rf $HOME/.cache/pip

ADD ./ abcdrl
WORKDIR abcdrl
