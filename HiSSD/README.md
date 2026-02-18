# HiSSD and ODIS Baselines

We modify the original HiSSD codebase for running our experiments on the ODIS and HiSSD baselines

## Installation

To install the core HiSSD codebase please refer the original codebase's documentation found in `README-HiSSD.md`. Refer to the `debugging.md` file if problems are encountered.

Need to set the following environment variables to run train and evaluation experiments
```
"LOG_DIR_PREFIX": "path/to/logs",
"DATASET_DIR_PREFIX":"path/to/datasets/",
"SC2PATH": "path/to/install/StarCraftII",
```

### Training the HiSSD and ODIS baselines

To run training for the HiSSD baseline on the `terran_5_vs_5` task, please run the following: 
```bash
python main.py --mto --config=hissd --env-config=sc2_offline_smacv2 --task-config=terran_5_vs_5-expert-limited-data  --offline_data_folder=path/to/hissd_datasets --seed=100
```

To run training for the ODIS baseline on the `terran_5_vs_5` task, please run the following: 
```bash 
python main.py --mto --config=odis --env-config=sc2_offline_smacv2 --task-config=terran_5_vs_5-expert-limited-data --offline_data_folder=path/to/hissd_datasets --seed=100
```

The above can be done for the `protoss_5_vs_5` task by the changing the config parameter to `--task-config=protoss_5_vs_5-expert-limited-data`

### Evaluating the trained HiSSD and ODIS baselines 

To run evaluation for trained HiSSD models:
```bash 
python main.py --mto --config=hissd --env-config=sc2_offline_smacv2 --task-config=terran_5_vs_5-expert  --seed=100 --test_nepisode=100 --evaluate=True --load_step=30050000 --checkpoint_path=/path/to/models/
```

To run evaluation for trained ODIS models: 

```bash 
python main.py --mto --config=odis --env-config=sc2_offline_smacv2 --task-config=terran_5_vs_5-expert --seed=100 --test_nepisode=100 --evaluate=True --load_step=30050000 --checkpoint_path=/path/to/models/
```
