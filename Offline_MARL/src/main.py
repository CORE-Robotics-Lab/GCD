import numpy as np
import os
import collections
import collections.abc
from os.path import dirname, abspath, join
from copy import deepcopy
from sacred import Experiment, SETTINGS
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds
import sys
import torch as th
from utils.logging import get_logger
import yaml
from waves import WaveData
from share_array import make_shared_array, get_shared_array

from run import REGISTRY as run_REGISTRY
SETTINGS['CAPTURE_MODE'] = "fd" # set to "no" if you want to see stdout/stderr in console
logger = get_logger()

ex = Experiment("pymarl", save_git_info=False)
ex.logger = logger
ex.captured_out_filter = apply_backspaces_and_linefeeds

results_path = join(dirname(dirname(abspath(__file__))), "results")


@ex.main
def my_main(_run, _config, _log):
    # Setting the random seed throughout the modules
    config = config_copy(_config)
    seed = config["seed"]
    print(f"seed is {seed }")
    np.random.seed(config["seed"])
    th.manual_seed(config["seed"])
    config['env_args']['seed'] = config["seed"]

    # run
    if "use_per" in _config and _config["use_per"]:
        run_REGISTRY['per_run'](_run, config, _log)
    # elif 'use_offline' in _config and _config['use_offline']:
    #     run_REGISTRY['offline_run'](_run, config, _log)
    elif 'mpe_env' in _config['env']:
        run_REGISTRY['mpe_run'](_run, config, _log)
    else:
        run_REGISTRY[_config['run']](_run, config, _log)

def _get_config(params, arg_name, subfolder):
    config_name = None
    for _i, _v in enumerate(params):
        if _v.split("=")[0] == arg_name:
            config_name = _v.split("=")[1]
            del params[_i]
            break

    if config_name is not None:
        with open(os.path.join(os.path.dirname(__file__), "config", subfolder, "{}.yaml".format(config_name)), "r") as f:
            try:
                config_dict = yaml.load(f,Loader=yaml.FullLoader)
            except yaml.YAMLError as exc:
                assert False, "{}.yaml error: {}".format(config_name, exc)
        return config_dict
    
def update_param(params, key, value):
    found = False
    for i, p in enumerate(params):
        if p.startswith(key + "="):
            params[i] = f"{key}={value}"
            found = True
            break
    if not found:
        params.append(f"{key}={value}")
    return params


def recursive_dict_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def config_copy(config):
    if isinstance(config, dict):
        return {k: config_copy(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [config_copy(v) for v in config]
    else:
        return deepcopy(config)


def parse_command(params, key, default):
    result = default
    for _i, _v in enumerate(params):
        if _v.split("=")[0].strip() == key:
            result = _v[_v.index('=')+1:].strip()
            break
    return result


if __name__ == '__main__':
    params = deepcopy(sys.argv)

    # Get the defaults from default.yaml
    with open(os.path.join(os.path.dirname(__file__), "config", "default.yaml"), "r") as f:
        try:
            config_dict = yaml.load(f,Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            assert False, "default.yaml error: {}".format(exc)

    # Load algorithm and env base configs
    env_config = _get_config(params, "--env-config", "envs")
    alg_config = _get_config(params, "--config", "algs")
    # config_dict = {**config_dict, **env_config, **alg_config}
    config_dict = recursive_dict_update(config_dict, env_config)
    config_dict = recursive_dict_update(config_dict, alg_config)

    # now add all the config to sacred
    ex.add_config(config_dict)

    # Save to disk by default for sacred
    if 'map_name' in config_dict['env_args'].keys():
        map_name = parse_command(params, "env_args.map_name", config_dict['env_args']['map_name'])
        
        subteam = parse_command(params, "env_args.subteam", '')
        checkpoint_path = parse_command(params, "checkpoint_path", '')
        if subteam != '' and checkpoint_path != '':
            map_name = f"test_{map_name}_{subteam}"
            params = update_param(params, "env_args.map_name", map_name)
    else:
        map_name = config_dict['env']

    if "ocean" in config_dict['env_args']['map_name']:
        WaveDataLoader = WaveData()
        whole_intensity_data = WaveDataLoader.get_region_intensity_all_data(config_dict['env_args']['scenario_difficulty'])
        make_shared_array(whole_intensity_data, name='wave_info')

    algo_name = parse_command(params, "name", config_dict['name']) 
    file_obs_path = join(results_path, "sacred", map_name, algo_name)
    
    logger.info("Saving to FileStorageObserver in {}.".format(file_obs_path))
    ex.observers.append(FileStorageObserver.create(file_obs_path))

    ex.run_commandline(params)

    # flush
    sys.stdout.flush()
