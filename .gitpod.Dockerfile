FROM gitpod/workspace-python:2022-11-07-20-39-06
USER gitpod
RUN if ! grep -q "export PIP_USER=no" "$HOME/.bashrc"; then printf '%s\n' "export PIP_USER=no" >> "$HOME/.bashrc"; fi

RUN sudo apt-get -y install wget unzip software-properties-common \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev patchelf swig

ARG REQ_DIR=requirements/requirements-dev.txt
RUN echo REQ_DIR=$REQ_DIR
ADD $REQ_DIR requirements.txt

RUN pip install -r requirements.txt
