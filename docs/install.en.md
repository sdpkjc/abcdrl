# Installation 🛠

[![Open in Gitpod](https://gitpod.io/button/open-in-gitpod.svg)](https://gitpod.io/#https://github.com/sdpkjc/abcdrl)

## `Pip`

=== "CPU"

    ```bash
    # 0. Prerequisites: Python3.8+ & Pip
    # 1. Pull git repository from github
    git clone https://github.com/sdpkjc/abcdrl.git && cd abcdrl
    # 2. Install dependencies
    pip install -r requirements/requirements.txt
    # 3. Run the DQN algorithm
    python abcdrl/dqn.py
    ```

=== "GPU"

    ```bash
    # 0. Prerequisites: Conda & Nvidia Driver
    # 1. Pull git repository from github
    git clone https://github.com/sdpkjc/abcdrl.git && cd abcdrl
    # 2. Create and activate the virtual environment
    conda create -n abcdrl python=3.9 pip && conda activate abcdrl
    # 3. Install cudatoolkit and the corresponding version of Pytorch
    conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge
    # 4. Install dependencies
    pip install -r requirements/requirements.txt
    # 5. Run the DQN algorithm
    python abcdrl/dqn.py
    ```

    !!! note
        There are many ways to install pytorch, see [Mu Li's video tutorials](https://b23.tv/qvAxVzd) for details.

        Version selection of `cudatoolkit` is related to Nvidia Driver version, refer to [Mu Li's video tutorial](https://b23.tv/qvAxVzd) and [Pytorch installation page](https://pytorch.org/get-started/locally/).

## `Docker`

=== "CPU"

    ```bash
    # 0. Prerequisites: Docker
    # 1. Pull git repository from github
    git clone https://github.com/sdpkjc/abcdrl.git && cd abcdrl
    # 2. Build the docker image
    docker build . -t abcdrl
    # 3. Run the DQN algorithm
    docker run --rm abcdrl python abcdrl/dqn.py
    ```

=== "GPU"

    ```bash
    # 0. Prerequisites: Docker & Nvidia Drive & NVIDIA Container Toolkit
    # 1. Pull git repository from github
    git clone https://github.com/sdpkjc/abcdrl.git && cd abcdrl
    # 2. Build the docker image
    docker build . -t abcdrl
    # 3. Run the DQN algorithm
    docker run --rm --gpus all abcdrl python abcdrl/dqn.py
    ```

    !!! note
        Docker Container parameters and the detailed installation process of the NVIDIA Container Toolkit can be found here: [Nvidia Docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)

    !!! warning
        Build an image using our provided Dockerfile and train on GPU. Nvidia Driver needs to support CUDA11.7.

        Using the `nvidia-smi` command, look at the `CUDA Version: xx.x` in the top right corner. It should be 11.7 or greater.
