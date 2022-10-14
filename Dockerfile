FROM nvidia/cuda:11.6.2-runtime-ubuntu20.04

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
    libosmesa6-dev patchelf
RUN ln -s /usr/bin/python3 /usr/bin/python

RUN pip install poetry

COPY poetry.lock poetry.lock
COPY pyproject.toml pyproject.toml

RUN poetry install
RUN poetry install -E atari
RUN poetry install -E mujoco
