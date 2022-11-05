FROM gitpod/workspace-full:2022-11-04-17-43-13

RUN sudo apt-get -y install python3-pip
RUN sudo apt-get -y install wget unzip software-properties-common \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev patchelf

ARG REQ_DIR=requirements/requirements-dev.txt
RUN echo REQ_DIR=$REQ_DIR
ADD $REQ_DIR requirements.txt

RUN pip install -r requirements.txt &&\
    rm -rf $HOME/.cache/pip

ADD ./ abcdrl

WORKDIR abcdrl
