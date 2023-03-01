FROM nvidia/cuda:11.7.0-runtime-ubuntu22.04
LABEL maintainer="hi@sdpkjc.com"

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="$HOME/.pyenv/bin:$HOME/.pyenv/shims:$PATH"
ENV PYENV_ROOT="$HOME/.pyenv"

ARG PYTHON_VERSION=3.9
RUN apt-get update && apt-get -y install \
    python3-pip git unzip \
    software-properties-common \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev patchelf swig \
    make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev \
    wget curl llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev && \
    apt-get autoclean && rm -rf /var/lib/apt/lists/* && \
    curl -fsSL https://pyenv.run | bash && \
	pyenv update && \
	pyenv install ${PYTHON_VERSION} && \
	pyenv global ${PYTHON_VERSION} && \
	for exec in global; do printf '%s\n' 'source "$HOME/.gp_pyenv.d/userbase.bash"' >> "$PYENV_ROOT/libexec/pyenv-$exec"; done && \
	python3 -m pip install --no-cache-dir --upgrade pip

RUN pip install torch==1.13.0 torchvision torchaudio
ARG REQ_DIR=requirements/requirements-dev.txt
ADD $REQ_DIR requirements.txt
RUN pip install -r requirements.txt && \
    rm -rf $HOME/.cache/pip

ADD ./ abcdrl
WORKDIR abcdrl
