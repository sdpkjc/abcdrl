# 安装

## pip

=== "CPU"

    ```bash
    # 0. 安装 python3.8+ & pip
    # 1. 拉取代码
    git clone https://e.coding.net/sdpkjc/abcdrl/abcdrl.git && cd abcdrl
    # 2. 安装依赖
    pip install -r requirements/requirements.txt
    # 3. 运行 DQN 算法
    python abcdrl/dqn.py
    ```

=== "GPU"

    ```bash
    # 0. 安装 Conda & Nvidia Driver
    # 1. 拉取代码
    git clone https://e.coding.net/sdpkjc/abcdrl/abcdrl.git && cd abcdrl
    # 2. 建立虚拟环境并激活
    conda create -n abcdrl python3.9 pip && conda activate abcdrl
    # 3. 安装 cudatoolkit 和对应版本的 Pytorch
    conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge
    # 4. 安装依赖
    pip install -r requirements/requirements.txt
    # 5. 运行 DQN 算法
    python abcdrl/dqn.py
    ```

    !!! note
        安装方法有多种可选，具体可参考视频 [李沐：环境安装，BERT、GPT、T5 性能测试，和横向对比【100亿模型计划】-哔哩哔哩](https://b23.tv/qvAxVzd) 第二部分。

        `cudatoolkit` 的版本选择与 Nvidia Driver 版本相关，请参考[视频教程](https://b23.tv/qvAxVzd)和 [Pytorch 官网安装页面](https://pytorch.org/get-started/locally/)。

!!! note
    若您的 Nvidia Driver 和 GPU 设备支持 CUDA10.2 版本，且已安装 cudatoolkit10.2，直接使用上述的 CPU 安装流程即可。



## Docker

=== "CPU"

    ```bash
    # 0. 安装 Docker
    curl https://get.docker.com | sh \
    && sudo systemctl --now enable docker
    # 1. 使用我们在 Docker Hub 上的镜像，运行 DQN 算法。
    docker run --rm sdpkjc/abcdrl:latest python abcdrl/dqn.py
    ```

=== "GPU"

    ```bash
    # 以 Ubuntu 为例
    # 0. 安装 Nvidia Drive
    # 1. 安装 Docker
    curl https://get.docker.com | sh \
    && sudo systemctl --now enable docker
    # 2. 安装 NVIDIA Container Toolkit
    sudo apt-get update
    sudo apt-get install -y nvidia-docker2
    sudo systemctl restart docker
    # 3. 使用我们在 Docker Hub 上的镜像，运行 DQN 算法。
    docker run --rm --gpus all sdpkjc/abcdrl:latest python abcdrl/dqn.py
    ```

    !!! note
        Docker image 参数和 NVIDIA Container Toolkit 详细安装过程可参考：[Nvidia Docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)

    !!! warning
        使用我们提供的 Docker 镜像并使用 GPU 训练，Nvidia Driver 需支持 CUDA11.6；使用 `nvidia-smi` 命令，查看右上角的 `CUDA Version: xx.x` 信息，大于等于 11.6 即可。
