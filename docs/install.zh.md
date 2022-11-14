# 安装

## `Pip`

=== "CPU"

    ```bash
    # 0. 安装 Python3.8+ & Pip
    # 1. 拉取代码仓库
    git clone https://github.com/sdpkjc/abcdrl.git && cd abcdrl
    # 2. 安装依赖
    pip install -r requirements/requirements.txt
    # 3. 运行 DQN 算法
    python abcdrl/dqn.py
    ```

=== "GPU"

    ```bash
    # 0. 安装 Conda & Nvidia Driver
    # 1. 拉取代码仓库
    git clone https://github.com/sdpkjc/abcdrl.git && cd abcdrl
    # 2. 建立虚拟环境并激活
    conda create -n abcdrl python=3.9 pip && conda activate abcdrl
    # 3. 安装 cudatoolkit 和对应版本的 Pytorch
    conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge
    # 4. 安装依赖
    pip install -r requirements/requirements.txt
    # 5. 运行 DQN 算法
    python abcdrl/dqn.py
    ```

    !!! note
        安装方法有多种可选，具体可参考视频 [李沐：环境安装，BERT、GPT、T5 性能测试，和横向对比【100亿模型计划】-哔哩哔哩](https://b23.tv/qvAxVzd) 。

        `cudatoolkit` 的版本选择与 Nvidia Driver 版本相关，请参考[视频教程](https://b23.tv/qvAxVzd)和 [Pytorch 官网安装页面](https://pytorch.org/get-started/locally/)。

## `Docker`

=== "CPU"

    ```bash
    # 0. 安装 Docker
    # 1. 拉取代码仓库
    git clone https://github.com/sdpkjc/abcdrl.git && cd abcdrl
    # 2. 构建 docker 镜像
    docker build . -t abcdrl
    # 3. 运行 DQN 算法。
    docker run --rm abcdrl python abcdrl/dqn.py
    ```

=== "GPU"

    ```bash
    # 0. 安装 Docker & Nvidia Drive & NVIDIA Container Toolkit
    # 1. 拉取代码仓库
    git clone https://github.com/sdpkjc/abcdrl.git && cd abcdrl
    # 2. 构建 docker 镜像
    docker build . -t abcdrl
    # 3. 运行 DQN 算法。
    docker run --rm --gpus all abcdrl python abcdrl/dqn.py
    ```

    !!! note
        Docker 容器参数和 NVIDIA Container Toolkit 详细安装过程可参考：[Nvidia Docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)

    !!! warning
        使用我们提供的 Dockerfile 构建镜像并使用 GPU 训练，Nvidia Driver 需支持 CUDA11.7；

        使用 `nvidia-smi` 命令，查看右上角的 `CUDA Version: xx.x` 信息，大于等于 11.7 即可。
