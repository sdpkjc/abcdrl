FROM nvidia/cuda:11.7.0-runtime-ubuntu22.04
LABEL maintainer="pazyx728@gmail.com" \
      version="v0.1"

ENV DEBIAN_FRONTEND=noninteractive
RUN sed -i "s@http://.*archive.ubuntu.com@https://mirrors.tuna.tsinghua.edu.cn@g" /etc/apt/sources.list && \
    sed -i "s@http://.*security.ubuntu.com@https://mirrors.tuna.tsinghua.edu.cn@g" /etc/apt/sources.list
RUN apt-get update
RUN apt-get -y install python3-pip
RUN apt-get -y install wget unzip software-properties-common \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev patchelf swig
RUN ln -s /usr/bin/python3 /usr/bin/python

RUN pip install torch==1.13.0 torchvision torchaudio -i https://pypi.tuna.tsinghua.edu.cn/simple

ARG REQ_DIR=requirements/requirements-dev.txt
RUN echo REQ_DIR=$REQ_DIR
ADD $REQ_DIR requirements.txt

RUN pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple &&\
    rm -rf $HOME/.cache/pip

ADD ./ abcdrl

WORKDIR abcdrl
