# Graph Conditoned Diffusion (GCD)

We provide the necessary code and scripts for running our model Graph Conditioned Diffusion, and the baseline model MADiff.

## Installation 

```
conda create -n madiff python=3.8.18
conda activate madiff
pip3 install torch

pip install -r requirements.txt  # TODO: come back and edit these to only required packages

pip install -e ./third_party/og-marl

pip install git+https://github.com/oxwhirl/smac.git
pip install -r ./third_party/og-marl/install_environments/requirements/smacv1.txt
cd ./third_party/og-marl/install_environments/
./smacv1.sh

cd ~/StarCraftII/Maps/
wget https://github.com/oxwhirl/smacv2/releases/download/maps/SMAC_Maps.zip


pip install git+https://github.com/oxwhirl/smacv2.git

```

Install the [Deep Graph Library](https://www.dgl.ai/pages/start.html)


### Environment Variables
Need to set the following environment variables to run train and evaluation experiments

```
"LOG_DIR_PREFIX": "path/to/logs",
"DATASET_DIR_PREFIX":"path/to/datasets/",
"SC2PATH": "path/to/install/StarCraftII",
```


### smac_v2 install modification 
need to add the following to smacv2 package:

Change the class FixedDistribution to the following:

```python
class FixedDistribution(Distribution):
    """A generic disribution that draws from a fixed list.
    May operate in test mode, where items are drawn sequentially,
    or train mode where items are drawn randomly. Example uses of this
    are for team generation or per-agent accuracy generation in SMAC by
    drawing from separate fixed lists at test and train time.
    """

    def __init__(self, config):
        """
        Args:
            config (dict): Must contain `env_key`, `test_mode` and `items`
            entries. `env_key` is the key to pass to the environment so that it
            recognises what to do with the list. `test_mode` controls the sampling
            behaviour (sequential if true, uniform at random if false), `items`
            is the list of items (team configurations/accuracies etc.) to sample from.
        """
        self.config = config
        self.env_key = config["env_key"]
        self.test_mode = config["test_mode"]
        self.teams = config["items"]
        self.index = 0

    def generate(self) -> Dict[str, Dict[str, Any]]:
        """Returns:
        Dict: Returns a dict of the form
        {self.env_key: {"item": <item>, "id": <item_index>}}
        """
        if self.test_mode:
            team = self.teams[self.index]
            team_id = self.index
            self.index = (self.index + 1) % len(self.teams)
            shuffle(team)
        
            return {self.env_key: {"ally_team": team, "enemy_team": team, "id": team_id,}}
    
        else:
            team = choice(self.teams)
            team_id = self.teams.index(team)
            shuffle(team)
            return {self.env_key: {"item": team, "id": team_id}}

    @property
    def n_tasks(self):
        return len(self.teams)

```


## Running Experiments

In this section we describe how to run training and evaluation experiments presented in the paper.

### Training GCD Models and MADiff

To train the **GCD-Homo** model, please run the following command:
```bash
python run_scripts/train.py -e exp_specs/train/mad_hetgat_smac_v2_terran_5_vs_5.yaml --job_name "{dataset}/{model}-graph_cond_{graph_condition}" --graph_condition True --model models.SharedConvGCNDeconvFixed --dataset terran_5_vs_5-New --seed 100 --blind_own_type False
```
 
To train the **GCD-HetGCN** model, please run the following command:
```bash
python run_scripts/train.py -e exp_specs/train/mad_hetgat_smac_v2_terran_5_vs_5.yaml --job_name "{dataset}/{model}-graph_cond_{graph_condition}" --graph_condition True --model models.SharedConvHetGCNDeconvFixed --dataset terran_5_vs_5-New --seed 100 --blind_own_type False
```

To train the **GCD-HetGAT** model, please run the following command:
```bash
python run_scripts/train.py -e exp_specs/train/mad_hetgat_smac_v2_terran_5_vs_5.yaml --job_name "{dataset}/{model}-graph_cond_{graph_condition}" --graph_condition True --model models.SharedConvHetGATDeconvFixed --dataset terran_5_vs_5-New --seed 100 --blind_own_type False
```

To train the **MADiff** model, please run the following command:
```bash
python run_scripts/train.py -e exp_specs/train/mad_smac_v2_terran_5_vs_5.yaml --job_name "{dataset}/{model}-decent" --dataset terran_5_vs_5-New --seed 100 --blind_own_type False --decentralized_execution True
```

### Evaluating the trained GCD Models and MADiff

In our paper, we run 100 evaluation episodes on the trained models across all team compositions. With the following command, you can run evaluation on a specific team composition. In these examples, we evluate on the team composition with the label "2ma_2mr_1md". All team composition labels can be found in the paper. 

In the following the `--log_dir` argument should point to the directory where the trained model checkpoints are stored. For example to evaluate GCD-HetGAT, the directory would take the following form `logs/mad_smac_v2/terran_5_vs_5-New/models.SharedConvHetGATDeconvFixed-graph_cond_True/300`. By changing this direcrtory, you can evaluate different trained models. You can specify the output directory for evaluation results using the `--exp_name` argument.

To evaluate the **GCD-HetGAT** model, please run the following command:
```bash
python run_scripts/evaluate_smac_v2_subteams.py -e exp_specs/eval/eval_madiff_hetgat_terran_5_vs_5_subteam.yaml --exp_name "mad_hetgat_graph_cond_decentralized/terran_5_vs_5_new" --log_dir "path/to/log/dir" --subteam "2ma_2mr_1md" --load_step 500000 --decentralized_execution True --blind_own_type False
```

#### Training Ablation Results 
The models in our ablation study can be trained using the same commands as above, but changing the `--blind_own_type True` flag to `True`. This can be achieved similarly for the evaluation commands.










## Acknowledgements
The codebase is built on the [MADiff repo](https://github.com/zbzhu99/madiff)


