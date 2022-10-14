FROM nvidia/cuda:11.6.2-runtime-ubuntu20.04
LABEL maintainer="pazyx728@gmail.com" \
      version="v0.1"

ENV DEBIAN_FRONTEND=noninteractive
RUN sed -i "s@http://.*archive.ubuntu.com@https://mirrors.tuna.tsinghua.edu.cn@g" /etc/apt/sources.list && \
    sed -i "s@http://.*security.ubuntu.com@https://mirrors.tuna.tsinghua.edu.cn@g" /etc/apt/sources.list
RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         cmake \
         git \
         curl \
         ca-certificates \
         libgl1-mesa-dev \
         libgl1-mesa-glx \
         libglew-dev \
         libosmesa6-dev patchelf && \
     rm -rf /var/lib/apt/lists/*

RUN curl -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh

RUN echo "channels:\n\
  - conda-forge\n\
  - defaults\n\
show_channel_urls: true\n\
default_channels:\n\
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main\n\
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r\n\
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2\n\
custom_channels:\n\
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud\n\
  msys2: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud\n\
  bioconda: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud\n\
  menpo: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud\n\
  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud\n\
  pytorch-lts: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud\n\
  simpleitk: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud" > /root/.condarc

ENV PATH /opt/conda/bin:$PATH
RUN /opt/conda/bin/conda install python=3.9 pip && \
    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

ARG REQ_DIR=requirements/requirements-dev.yaml
RUN echo REQ_DIR=$REQ_DIR
ADD $REQ_DIR requirements.yaml
RUN /opt/conda/bin/conda env create -f requirements.yaml && \
    /opt/conda/bin/conda clean -ya

RUN rm -rf $HOME/.cache/pip

ARG ENV_NAME=abcdrl_dev
RUN echo ENV_NAME=$ENV_NAME
RUN echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate "$ENV_NAME >> ~/.bashrc
SHELL ["/bin/bash", "-c"]
