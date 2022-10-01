# Bulid Env

```bash
conda env create --name rl_a --file ./requirements.yaml

pip install mujoco-py

mkdir ~/.mujoco
wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O ~/.mujoco/mujoco210-linux-x86_64.tar.gz
tar -zxvf ~/.mujoco/mujoco210-linux-x86_64.tar.gz
mv ./mujoco210 ~/.mujoco/

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia:~/.mujoco/mujoco210/bin
```

## F

```bash
sudo apt-get update -y
sudo apt-get install libosmesa6-dev
sudo apt-get install libglew-dev glew-utils
sudo apt-get -y install patchelf
```

```bash
conda config --add channels conda-forge
```
