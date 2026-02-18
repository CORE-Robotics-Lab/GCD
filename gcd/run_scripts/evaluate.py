
import argparse
import os



import diffuser.utils as utils
import yaml
from diffuser.utils.launcher_util import build_config_from_dict, build_nested_variant_generator

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def evaluate(Config):
    evaluator = None
    Config.condition_guidance_w = getattr(Config, "condition_guidance_w", None)
    
    log_dir_prefix = os.environ.get('LOG_DIR_PREFIX')
    
    for load_step in Config.load_steps:
        ckpt_file_path = os.path.join(log_dir_prefix, Config.log_dir, f"checkpoint/state_{load_step}.pt")
        
        if not os.path.exists(ckpt_file_path):
            print(f"Checkpoint file {ckpt_file_path} not found. Skipping evaluation.")
            continue
        
        if evaluator is None:
            evaluator_config = utils.Config(Config.evaluator, verbose=False)
            evaluator = evaluator_config()
            
            evaluator.init(eval_experiment_name=Config.exp_name, 
                           log_dir=f'{log_dir_prefix}/{Config.log_dir}',
                           eval_env=Config.eval_env, 
                           num_eval=Config.num_eval,
                           eval_env_subteam=getattr(Config, "eval_env_subteam", ''),
                           num_envs=getattr(Config, "num_envs", Config.num_eval),
                           condition_guidance_w=getattr(Config, "condition_guidance_w", None), 
                           use_ddim_sample=Config.use_ddim_sample,
                           n_ddim_steps=Config.n_ddim_steps,)
            
        evaluator.evaluate(load_step=load_step)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-e", "--experiment", help="experiment specification file")

    args = parser.parse_args()
    
    with open(args.experiment, "r") as spec_file:
        spec_string = spec_file.read()
        exp_specs = yaml.load(spec_string, Loader=yaml.SafeLoader)
    
    vg_fn = build_nested_variant_generator(exp_specs)
    for variant in vg_fn():
        Config = build_config_from_dict(variant)        
        evaluate(Config)

