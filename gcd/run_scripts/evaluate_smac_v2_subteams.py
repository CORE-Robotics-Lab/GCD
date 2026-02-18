
import argparse
import os



import diffuser.utils as utils
import yaml

from diffuser.utils.launcher_util import (
    build_config_from_dict, 
    build_nested_variant_generator, 
    set_config_override_args,
    update_exp_spec_dict
)


import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def evaluate(Config):
    
    evaluator = None
    Config.condition_guidance_w = getattr(Config, "condition_guidance_w", None)

    assert Config.eval_env_subteam != '', "Config.eval_env_subteam is not specified"

    log_dir_prefix = os.environ.get('LOG_DIR_PREFIX')
    
    
    ckpt_file_path = os.path.join(log_dir_prefix, Config.log_dir, f"checkpoint/state_{Config.load_step}.pt")
    
    if not os.path.exists(ckpt_file_path):
        print(f"Checkpoint file {ckpt_file_path} not found. Skipping evaluation.")
        # continue
    
    if evaluator is None:
    
        evaluator_config = utils.Config(Config.evaluator, verbose=True)
        evaluator = evaluator_config()
        
        evaluator.init(eval_experiment_name=Config.exp_name, 
                        log_dir=f'{log_dir_prefix}/{Config.log_dir}',
                        eval_env=Config.eval_env, 
                        eval_env_subteam=Config.eval_env_subteam,
                        num_eval=Config.num_eval,
                        num_envs=getattr(Config, "num_envs", Config.num_eval),
                        condition_guidance_w=Config.condition_guidance_w, 
                        use_ddim_sample=Config.use_ddim_sample, 
                        n_ddim_steps=Config.n_ddim_steps,
                        eval_freq=Config.eval_freq,
                        prob_obs_enemy=Config.prob_obs_enemy, 
                        decentralized_execution=Config.decentralized_execution,
                        )
            
    evaluator.evaluate(load_step=Config.load_step)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-e", "--experiment", help="experiment specification file")
    
    parser.add_argument("--exp_name", help="smacv2 subteam", type=str, default='')
    parser.add_argument("--log_dir", help="smacv2 subteam", type=str, default='')
    parser.add_argument("--subteam", help="smacv2 subteam", type=str, default='')
    
    
    args, override_args = parser.parse_known_args()

    # args = parser.parse_args()

    with open(args.experiment, "r") as spec_file:
        spec_string = spec_file.read()
        exp_specs = yaml.load(spec_string, Loader=yaml.SafeLoader)
    
    Config = build_config_from_dict(exp_specs)
    Config, override_args_dict = set_config_override_args(Config, override_args)
    # exp_specs = update_exp_spec_dict(exp_specs, override_args_dict)

    Config.exp_name = getattr(args, "exp_name", '')
    Config.log_dir = getattr(args, "log_dir", '')
    Config.eval_env_subteam = getattr(args, "subteam", '')

    evaluate(Config)