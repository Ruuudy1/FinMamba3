# Drama: Mamba-Enabled Model-Based Reinforcement Learning Is Sample and Parameter Efficient

This repository provides an implementation of [Drama](https://openreview.net/forum?id=7XIkRgYjK3&nesting=2&sort=date-desc): a Mamba/Mamba2 powered model based reinforcement learning agent.

If you find Drama useful, please reference in your paper:
```
@inproceedings{
    wang2025drama,
    title={Drama: Mamba-Enabled Model-Based Reinforcement Learning Is Sample and Parameter Efficient},
    author={Wenlong Wang and Ivana dusparic and Yucheng Shi and Ke Zhang and Vinny Cahill},
    booktitle={The Thirteenth International Conference on Learning Representations},
    year={2025},
    url={https://openreview.net/forum?id=7XIkRgYjK3}
}
```


## Training and Evaluating Instructions
### Requirements

- **Python**: 3.10
- **Operating System**: Ubuntu 22.04 recommended (for Windows, use Docker)

### Setup Instructions

1. Create and activate a Conda environment:
```
conda create --name drama python=3.10
conda activate drama
```
2. Note that because `mamba-ssm` in _requirements.txt_ requires `pytorch`, so one should install `pytorch` before _requirements.txt_.
```
pip install torch==2.2.1
pip install -r requirements.txt
```
### Docker Instructions
---

1. Build the Docker image
```
docker build -f Dockerfile -t drama .
```
2. Run the container with GPU support
```
docker run --gpus all -it --rm drama
```
The container starts inside `src/` automatically. Run `python train.py` or `python eval.py` directly.

_Note: Running via Docker may result in slower performance. Please refer [here](https://forums.docker.com/t/docker-extremely-slow-on-linux-and-windows/129752), it is recommended to reproduce the result in ubuntu OS._

### Google Colab Instructions
---

Two ready-to-run notebooks are provided. Both require a GPU runtime (T4 or higher).

#### Option A: Evaluate a pretrained checkpoint (recommended for getting started)

Pretrained weights trained on ALE/Pong-v5 for 100k steps are available here:
[Google Drive - drama_checkpoints](https://drive.google.com/drive/folders/1jeoY7pMU8brPPiYDHzu3TOzhXSMdkAJy?usp=sharing)

1. Download `world_model.pth` and `agent.pth` from the Drive folder above
2. Upload them to your Google Drive under `My Drive/drama_checkpoints/`
3. Open `drama_colab_A100_Eval.ipynb` in Google Colab
4. Select `Runtime -> Change runtime type -> GPU`
5. Run cells 1-8 in order

The notebook will clone the repository, install all dependencies, apply compatibility patches, restore the checkpoints from your Drive, and run evaluation.

#### Option B: Train from scratch

1. Open `drama_colab_A100_Train.ipynb` in Google Colab
2. Select `Runtime -> Change runtime type -> A100 GPU` (requires Colab Pro)
3. Run cells 1-8 in order
4. Once training starts (cell 8), open a second browser tab connected to the same runtime and run cell 9 to back up checkpoints to Google Drive every 10 minutes

Training takes approximately 16 hours on an A100 for 100k steps on ALE/Pong-v5.

_Note: The install cells compile CUDA extensions for `causal-conv1d` and `mamba-ssm`. Do not skip them and do not change the install order._

### Training Instructions
---
All source code lives in the `src/` directory. Train with the default hyperparameters (the configuration file can be found at `src/config_files/configure.yaml`):
```
cd src
python train.py
```
If one wants to change the hyperparameter there are two ways:

1. Edit the configuration file `src/config_files/configure.yaml`.
2. Run `train.py` with parameters corresponding to the config. e.g., `python train.py --Models.WorldModel.Backbone Mamba`.

### Important parameters:
Drama supports three different dynamic models: _Transformer_, _Mamba_ and _Mamba-2_. It supports two type of behaviour models: _Actor-critic_ and _PPO_.


## Code references
We've referenced several other projects during the development of this code:
- [Mamba/Mamba-2](https://github.com/state-spaces/mamba)
- [STORM](https://github.com/weipu-zhang/STORM) 
- [DreamerV3](https://github.com/danijar/dreamerv3)
