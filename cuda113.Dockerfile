FROM nvidia/cuda:11.3.1-runtime-ubuntu18.04
LABEL maintainer="pazyx728@gmail.com"

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="$HOME/.pyenv/bin:$HOME/.pyenv/shims:$PATH"
ENV PIPENV_VENV_IN_PROJECT=true
ENV PYENV_ROOT="$HOME/.pyenv"

RUN apt-get update && apt-get -y install \
    python3-pip wget unzip \
    git software-properties-common \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev patchelf swig && \
    apt-get autoclean && rm -rf /var/lib/apt/lists/* && \
    curl -fsSL https://pyenv.run | bash && \
	pyenv update && \
	pyenv install ${PYTHON_VERSION} && \
	pyenv global ${PYTHON_VERSION} && \
	for exec in global; do printf '%s\n' 'source "$HOME/.gp_pyenv.d/userbase.bash"' >> "$PYENV_ROOT/libexec/pyenv-$exec"; done && \
	python3 -m pip install --no-cache-dir --upgrade pip && \
	pip install --no-cache-dir --upgrade

RUN pip install torch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
ARG REQ_DIR=requirements/requirements-dev.txt
ADD $REQ_DIR requirements.txt
RUN pip install -r requirements.txt && \
    rm -rf $HOME/.cache/pip

ADD ./ abcdrl
WORKDIR abcdrl
