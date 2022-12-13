FROM nvidia/cuda:11.7.0-runtime-ubuntu22.04
LABEL maintainer="pazyx728@gmail.com"

RUN apt-get update && apt-get -y install \
    python3-pip wget unzip \
    git software-properties-common \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev patchelf swig
RUN ln -s /usr/bin/python3 /usr/bin/python

RUN pip install torch==1.13.0 torchvision torchaudio
ARG REQ_DIR=requirements/requirements-dev.txt
ADD $REQ_DIR requirements.txt
RUN pip install -r requirements.txt && \
    rm -rf $HOME/.cache/pip

ADD ./ abcdrl
WORKDIR abcdrl

RUN echo -e "Welcome to abcdrl!\n\t Branch: $SOURCE_BRANCH\n\t Commit: $SOURCE_COMMIT\n\t Image: $IMAGE_NAME" > HELLO
CMD cat HELLO
