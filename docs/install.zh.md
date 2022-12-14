# å®è£ ð 

!!! example "å¿«éå¼å§"
    å¨ Gitpod ä¸­æå¼é¡¹ç®ï¼å¹¶ç«å³å¼å§ç¼ç ã

    [![Open in Gitpod](https://gitpod.io/button/open-in-gitpod.svg)](https://gitpod.io/#https://github.com/sdpkjc/abcdrl)

## `Docker`

=== "CPU"

    ```shell
    # 0. å®è£ Docker
    # 1. è¿è¡ DQN ç®æ³
    docker run --rm sdpkjc/abcdrl python abcdrl/dqn.py
    ```

=== "GPU"

    ```shell
    # 0. å®è£ Docker & Nvidia Drive & NVIDIA Container Toolkit
    # 1. è¿è¡ DQN ç®æ³
    docker run --rm --gpus all sdpkjc/abcdrl python abcdrl/dqn.py
    ```

    !!! note
        Docker å®¹å¨åæ°å NVIDIA Container Toolkit è¯¦ç»å®è£è¿ç¨å¯åèï¼[Nvidia Docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)

    !!! warning
        ä½¿ç¨æä»¬æä¾ç Docker éåå¹¶ä½¿ç¨ GPU è®­ç»ï¼Nvidia Driver éæ¯æ `CUDA11.7`ï¼

        ä½¿ç¨ `nvidia-smi` å½ä»¤ï¼æ¥çå³ä¸è§ç `CUDA Version: xx.x` ä¿¡æ¯ï¼å¤§äºç­äº 11.7 å³å¯ã

        å¦ææ¨çè®¾å¤ä»æ¯æ `CUDA11.3-11.6`ï¼å¯ä»¥ä½¿ç¨ `sdpkjc/abcdrl:cu113` éåãæ´èç `CUDA` çæ¬æä»¬ä¸åæä¾å®æ¹éåï¼å¯ä»¥åèæä»¬ç `Dockerfile` è¿è¡è°æ´æå»ºæ¨èªå·±çéåã

## `Pip`

=== "CPU"

    ```shell
    # 0. å®è£ Python3.8+ & Pip
    # 1. æåä»£ç ä»åº
    git clone https://github.com/sdpkjc/abcdrl.git && cd abcdrl
    # 2. å®è£ä¾èµ
    pip install -r requirements/requirements.txt
    # 3. è¿è¡ DQN ç®æ³
    python abcdrl/dqn.py
    ```

=== "GPU"

    ```shell
    # 0. å®è£ Conda & Nvidia Driver
    # 1. æåä»£ç ä»åº
    git clone https://github.com/sdpkjc/abcdrl.git && cd abcdrl
    # 2. å»ºç«èæç¯å¢å¹¶æ¿æ´»
    conda create -n abcdrl python=3.9 pip && conda activate abcdrl
    # 3. å®è£ cudatoolkit åå¯¹åºçæ¬ç Pytorch
    conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge
    # 4. å®è£ä¾èµ
    pip install -r requirements/requirements.txt
    # 5. è¿è¡ DQN ç®æ³
    python abcdrl/dqn.py
    ```

    !!! note
        Pytorch å®è£æ¹æ³æå¤ç§å¯éï¼å·ä½å¯åèè§é¢ [ææ²ï¼ç¯å¢å®è£ï¼BERTãGPTãT5 æ§è½æµè¯ï¼åæ¨ªåå¯¹æ¯ã100äº¿æ¨¡åè®¡åã-åå©åå©](https://b23.tv/qvAxVzd) ã

        `cudatoolkit` ççæ¬éæ©ä¸ Nvidia Driver çæ¬ç¸å³ï¼è¯·åè[è§é¢æç¨](https://b23.tv/qvAxVzd)å [Pytorch å®ç½å®è£é¡µé¢](https://pytorch.org/get-started/locally/)ã
