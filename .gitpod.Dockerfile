FROM gitpod/workspace-python:2022-11-07-20-39-06
USER gitpod
RUN if ! grep -q "export PIP_USER=no" "$HOME/.bashrc"; then printf '%s\n' "export PIP_USER=no" >> "$HOME/.bashrc"; fi

RUN apt-get update && apt-get -y install \
    python3-pip wget unzip \
    git software-properties-common \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev patchelf swig
RUN ln -s /usr/bin/python3 /usr/bin/python

ARG REQ_DIR=requirements/requirements-dev.txt
ADD $REQ_DIR requirements.txt
RUN pip install -r requirements.txt && \
    rm -rf $HOME/.cache/pip
