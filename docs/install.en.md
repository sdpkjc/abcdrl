# Installation 🛠

!!! example "Quickstart"
    Open the project in Gitpod and start coding immediately.

    [![Open in Gitpod](https://gitpod.io/button/open-in-gitpod.svg)](https://gitpod.io/#https://github.com/sdpkjc/abcdrl)

## `Docker`

=== "CPU"

    ```shell
    # 0. Prerequisites: Docker
    # 1. Run DQN algorithm
    docker run --rm sdpkjc/abcdrl python abcdrl/dqn_torch.py
    ```

=== "GPU"

    ```shell
    # 0. Prerequisites: Docker & Nvidia Drive & NVIDIA Container Toolkit
    # 1. Run DQN algorithm
    docker run --rm --gpus all sdpkjc/abcdrl python abcdrl/dqn_torch.py
    ```

    !!! note
        Docker Container parameters and the detailed installation process of the NVIDIA Container Toolkit can be found here: [Nvidia Docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker).

    !!! warning
        Using our docker image and train on GPU. Nvidia Driver needs to support `CUDA11.7`.

        Using `nvidia-smi` command, look at the `CUDA Version: xx.x` in the top right corner. It need to be 11.7 or greater.

        If your device only supports `CUDA11.3-11.6`, you can use `sdpkjc/abcdrl:cu113` image. For older `CUDA` versions, we don't officially support them, please refer to our `Dockerfile` to adjust and build your image.

## `Pip`

=== "CPU"

    ```shell
    # 0. Prerequisites: Python3.8+ & Pip
    # 1. Pull git repository from github
    git clone https://github.com/sdpkjc/abcdrl.git && cd abcdrl
    # 2. Install dependencies
    pip install -r requirements/requirements.txt
    # 3. Run DQN algorithm
    python abcdrl/dqn_torch.py
    ```

=== "GPU with PyTorch"

    ```shell
    # 0. Prerequisites: Conda & Nvidia Driver
    # 1. Pull git repository from github
    git clone https://github.com/sdpkjc/abcdrl.git && cd abcdrl
    # 2. Create and activate virtual environment
    conda create -n abcdrl python=3.9 pip && conda activate abcdrl
    # 3. Install cudatoolkit and the corresponding version of Pytorch
    conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge
    # 4. Install dependencies
    pip install -r requirements/requirements-torch.txt
    # 5. Run DQN algorithm
    python abcdrl/dqn_torch.py
    ```

    !!! note
        There are many ways to install pytorch, refer to [Mu Li's video tutorials](https://b23.tv/qvAxVzd) for details.

        Version selection of `cudatoolkit` is related to Nvidia Driver version, refer to [Mu Li's video tutorial](https://b23.tv/qvAxVzd) and [Pytorch installation page](https://pytorch.org/get-started/locally/).

=== "GPU with TensorFlow2"

    ```shell
    # 0. Prerequisites: Conda & Nvidia Driver
    # 1. Pull git repository from github
    git clone https://github.com/sdpkjc/abcdrl.git && cd abcdrl
    # 2. Create and activate virtual environment
    conda create -n abcdrl python=3.9 pip && conda activate abcdrl
    # 3. Install cudatoolkit
    conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
    # 4. Install dependencies
    pip install -r requirements/requirements-tf.txt
    # 5. Run DQN algorithm
    python abcdrl/dqn_tf.py
    ```

    !!! quote "Ref"
        [TensorFlow Official Guide](https://www.tensorflow.org/install/pip)
