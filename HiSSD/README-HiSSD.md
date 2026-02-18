# HiSSD: Learning Generalizable Skills from Offline Multi-Task Data for Multi-Agent Coordination

This is the implementation of the ICLR-2025 conference submission "Learning Generalizable Skills from Offline Multi-Task Data for Multi-Agent Coordination".

## Installation instructions

### Install StarCraft II

Set up StarCraft II and SMAC:

```bash
bash install_sc2.sh
```

This will download SC2.4.10 into the 3rdparty folder and copy the maps necessary to run over. You may also need to persist the environment variable `SC2PATH` (e.g., append this command to `.bashrc`):

```bash
export SC2PATH=[Your SC2 folder like /abc/xyz/3rdparty/StarCraftII]
```

### Install Python environment

Install Python environment with conda:

```bash
conda create -n hissd python=3.10 -y
conda activate hissd
pip install -r requirements.txt
```

### Configure SMAC package

The original [SMAC](https://github.com/oxwhirl/smac) package is extended by adding additional maps for multi-task evaluation. Here are a simple script to make some modifications in `smac` and copy additional maps to StarCraft II installation. Please make sure that you have set `SC2PATH` correctly.

```bash
git clone https://github.com/oxwhirl/smac.git
pip install -e smac/
bash install_smac_patch.sh
```

## Run experiments

You can execute the following command to run HiSSD with a toy task config, which will perform training on a small batch of data:

```bash
python src/main.py --mto --config=hissd --env-config=sc2_offline --task-config=toy --seed=1
```

The `--task-config` flag can be followed with any existing config name in the `src/config/tasks/` directory, and any other config named `xx` can be passed by `--xx=value`. 

As the dataset is large, we only contain the a toy task config of `3m` medium data in the `dataset` folder from the default code base. The data link to the full dataset is provided by this [Google Drive URL](https://drive.google.com/file/d/1BZSNaAzEN7nAGthsDCpIxXOo1oVoLdqP/view?usp=share_link) and you can substitute the original data with the full dataset. After putting the full dataset in `dataset` folder, you can reproduce the main results of HiSSD by running

```bash
# marine-hard expert
python src/main.py --mto --config=hissd --env-config=sc2_offline --task-config=marine-hard-expert --seed=1
# marine-hard medium
python src/main.py --mto --config=hissd --env-config=sc2_offline --task-config=marine-hard-medium --seed=1
# marine-hard medium-expert
python src/main.py --mto --config=hissd --env-config=sc2_offline --task-config=marine-hard-medium-expert --seed=1
# marine-hard medium-replay
python src/main.py --mto --config=hissd --env-config=sc2_offline --task-config=marine-hard-medium-replay --seed=1

```

All results will be stored in the `results` folder. You can see the console output, config, and tensorboard logging in the cooresponding directory.

## Build upon ODIS
Our HiSSD is built upon ODIS, please refer to [https://github.com/LAMDA-RL/ODIS](https://github.com/LAMDA-RL/ODIS).
