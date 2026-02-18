import gc
import multiprocessing
import os
import pickle
import sys
import time
import importlib

from collections import deque
from copy import deepcopy, copy
from typing import Optional

from einops import rearrange

import subprocess
from multiprocessing import Pipe, connection
from multiprocessing.context import Process

import numpy as np
import einops
import torch
from ml_logger import logger

import diffuser.utils as utils
from diffuser.utils.arrays import to_device, to_np, to_torch, apply_graphs
from diffuser.utils.launcher_util import build_config_from_dict


class MADEvaluatorWorker(Process):
    def __init__(self, 
                 parent_remote: connection.Connection, 
                 child_remote: connection.Connection,
                 queue: multiprocessing.Queue, 
                 verbose: bool = False,
                 ):
        
        self.parent_remote = parent_remote
        self.p = child_remote
        self.queue = queue
        self.initialized = False
        self.verbose = verbose
    
        self.debug_print_on = True 
        self.tb_logger = None
        
        super().__init__()


    def debug_print(self, message): 
        if self.debug_print_on: 
            subprocess.run(f'echo {message}', shell=True)
        else: 
            print(message)
        
    
    
    def _generate_samples(self, obs, returns, env_ts, ntypes=None, graphs=None, etypes=None):
        
        Config = self.Config

        env_ts = env_ts.clone()
        env_ts[torch.where(env_ts < 0)] = Config.max_path_length
        env_ts[torch.where(env_ts >= Config.max_path_length)] = Config.max_path_length

        attention_masks = np.zeros((obs.shape[0], Config.horizon + Config.history_horizon, Config.n_agents, 1))
        attention_masks[:, Config.history_horizon :] = 1.0

        shape = (obs.shape[0], Config.horizon + Config.history_horizon, *obs.shape[-2:],)  # b t a f
        
        if self.decentralized_execution and (not Config.use_hetgat_attention):
            # evaluation for decentralized MAD policies
            
            joint_cond_trajectories, joint_cond_masks, joint_attention_masks = ([], [], [],)
            
            for a_idx in range(Config.n_agents):
                local_cond_trajectories = np.zeros(shape, dtype=obs.dtype)
                local_cond_trajectories[:, :Config.history_horizon + 1, a_idx] = obs[:, :, a_idx]

                agent_mask = np.zeros(Config.n_agents)
                agent_mask[a_idx] = 1.0
                local_cond_masks = self.mask_generator(shape, agent_mask)

                local_attention_masks = copy(attention_masks)
                local_attention_masks[:, : Config.history_horizon, a_idx] = 1.0

                joint_cond_trajectories.append(to_torch(local_cond_trajectories, device=Config.device))
                joint_cond_masks.append(to_torch(local_cond_masks, device=Config.device))
                joint_attention_masks.append(to_torch(local_attention_masks, device=Config.device))

            joint_cond_trajectories = einops.rearrange(torch.stack(joint_cond_trajectories, dim=1), "b a ... -> (b a) ...")
            joint_cond_masks = einops.rearrange(torch.stack(joint_cond_masks, dim=1), "b a ... -> (b a) ...")
            joint_attention_masks = einops.rearrange(torch.stack(joint_attention_masks, dim=1), "b a ... -> (b a) ...")
            
            conditions = {"x": joint_cond_trajectories, "masks": joint_cond_masks,}
            
            returns = einops.repeat(returns, "b ... -> (b a) ...", a=Config.n_agents)
            env_ts = einops.repeat(env_ts, "b ... -> (b a) ...", a=Config.n_agents)

            time_to_sample_start = time.time() 
            joint_samples = self.trainer.ema_model.conditional_sample(conditions, 
                                                                      returns=returns, 
                                                                      env_ts=env_ts, 
                                                                      attention_masks=joint_attention_masks,
                                                                      )
            
            time_to_sample = time.time() - time_to_sample_start
            
            joint_samples = einops.rearrange(joint_samples, "(b a) ... -> b a ...", a=Config.n_agents)

            samples = []
            for a_idx in range(Config.n_agents):
                samples.append(joint_samples[:, a_idx, ..., a_idx, :])
            
            samples = torch.stack(samples, dim=-2)
            
        else:
            
            cond_trajectories = np.zeros(shape, dtype=obs.dtype)
            cond_trajectories[:, :Config.history_horizon + 1] = obs
            agent_mask = np.ones(Config.n_agents)
            cond_masks = self.mask_generator(shape, agent_mask)
        
            conditions = {
                "x": to_torch(cond_trajectories, device=Config.device),
                "masks": to_torch(cond_masks, device=Config.device),
            }
                
            attention_masks[:, :Config.history_horizon] = 1.0
            attention_masks = to_torch(attention_masks, device=Config.device)
            
            if Config.use_hetgat_attention:
                ntypes = to_torch(ntypes, device=Config.device)
                graphs = apply_graphs(graphs, device=Config.device)
                etypes = to_torch(etypes, device=Config.device)
        
            time_to_sample_start = time.time()    
            samples = self.trainer.ema_model.conditional_sample(conditions,
                                                                returns=returns,
                                                                env_ts=env_ts,
                                                                attention_masks=attention_masks,
                                                                ntypes=ntypes,
                                                                graphs=graphs,
                                                                etypes=etypes,
                                                                decentralized_execution=self.decentralized_execution, 
                                                                )

            time_to_sample = time.time() - time_to_sample_start
            
        
        
        samples = samples[:, Config.history_horizon:]
        
        return samples, time_to_sample


    def _evaluate(self, load_step: Optional[int] = None):
        
        assert (self.initialized is True), "Evaluator should be initialized before evaluation."
        
        Config = self.Config    
        loadpath = os.path.join(self.log_dir, "checkpoint")
        utils.set_seed(Config.seed)
        
        if Config.save_checkpoints:
            assert load_step is not None
            loadpath = os.path.join(loadpath, f"state_{load_step}.pt")
        else:
            loadpath = os.path.join(loadpath, "state.pt")

        # TODO: this all loads in model that was trained, what is needed to change to evaluate on a different environment?        
        state_dict = torch.load(loadpath, map_location=Config.device)
        state_dict["model"] = {k: v for k, v in state_dict["model"].items() 
                               if "value_diffusion_model." not in k}
        state_dict["ema"] = {k: v for k, v in state_dict["ema"].items() 
                             if "value_diffusion_model." not in k}

        self.trainer.step = state_dict["step"]    
        self.trainer.model.load_state_dict(state_dict["model"])
        self.trainer.ema_model.load_state_dict(state_dict["ema"])

        num_eval = Config.num_eval
        num_envs = Config.num_envs

        episode_rewards = []
        if Config.env_type == "smac" or Config.env_type == "smac_v2":
            episode_wins = []
            
        time_to_samples = [] # time taken to sample from the model

        cur_num_eval = 0
        

        while cur_num_eval < num_eval:
            
            num_episodes = min(num_eval - cur_num_eval, num_envs)
            rets = self._episodic_eval(num_episodes=num_episodes)
            episode_rewards.append(rets[1])
            
            if Config.env_type == "smac" or Config.env_type == "smac_v2":
                episode_wins.append(rets[2])
                
                avg_time_to_sample = rets[3]
                std_time_to_sample = rets[4]
                
                avg_time_to_step_smac = rets[5]
                std_time_to_step_smac = rets[6]
        

            cur_num_eval += num_episodes
            
            # print(f'episodes evaluated: {cur_num_eval}/{num_eval}')
        # print(f'evaluation time taken: {time.time() - start:.2f}s')
        
        episode_rewards = np.concatenate(episode_rewards, axis=0) 
        if Config.env_type == "smac" or Config.env_type == "smac_v2":
            episode_wins = np.concatenate(episode_wins, axis=0)
            # print(time_to_samples)
            # time_to_samples = np.concatenate(time_to_samples, axis=0)
            
        metrics_dict = dict(average_ep_reward=np.mean(episode_rewards, axis=0), 
                            std_ep_reward=np.std(episode_rewards, axis=0),)

        if Config.env_type == "smac" or Config.env_type == "smac_v2":
            metrics_dict["win_rate"] = np.mean(episode_wins)
            metrics_dict["avg_time_to_sample"] = avg_time_to_sample
            metrics_dict["std_time_to_sample"] = std_time_to_sample

            metrics_dict["avg_time_to_step_smac"] = avg_time_to_step_smac
            metrics_dict["std_time_to_step_smac"] = std_time_to_step_smac            




        logger.print(", ".join([f"{k}: {v}" for k, v in metrics_dict.items()]), color="green",)
        
        log_dir_prefix = os.environ.get('LOG_DIR_PREFIX')
        
        if Config.eval_freq == 0: # post-training evaluation
            if self.eval_env_subteam:
                save_file_path = (f"{log_dir_prefix}/eval_logs/{self.eval_experiment_name}/{Config.seed}/subteam_{self.eval_env_subteam}-step_{load_step}-ep_{num_eval}-ddim.json" 
                                  if getattr(Config, "use_ddim_sample", False) else f"results/step_{load_step}-ep_{num_eval}.json")
                
            else:
                save_file_path = (f"{log_dir_prefix}/eval_logs/{self.eval_experiment_name}/{Config.seed}/step_{load_step}-ep_{num_eval}-ddim.json" 
                                  if getattr(Config, "use_ddim_sample", False) else f"results/step_{load_step}-ep_{num_eval}.json")
    
        else: # save evaluation results during training
            save_file_path = (f"{self.log_dir}/results/step_{load_step}-ep_{num_eval}-ddim.json" 
                            if getattr(Config, "use_ddim_sample", False) else f"results/step_{load_step}-ep_{num_eval}.json")
    
        
        if self.rewrite_cgw:
            save_file_path = save_file_path.replace(".json", f"-cg_{self.trainer.ema_model.condition_guidance_w}.json")
    
        print(f"Saving evaluation results to {save_file_path}")
        logger.save_json({k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in metrics_dict.items()},
                         save_file_path,)
                    
        return metrics_dict
        
    
    def _update_return_to_go(self, rtg, reward):
        rtg = rtg * self.Config.returns_scale
        reward = torch.tensor(reward, device=rtg.device, dtype=rtg.dtype).reshape(1, -1)
        rtg = (rtg - reward) / self.Config.discount
        rtg = rtg / self.Config.returns_scale
        return rtg


    def _episodic_eval(self, num_episodes: int):
        """Evaluate for one episode each environment."""

        # `num_episodes` can be smaller than total number of environment, and
        # we only use the first `num_episodes` environments.
        assert (num_episodes <= self.Config.num_envs), f"num_episodes should be <= num_envs, but {num_episodes} > {self.Config.num_envs}"

        Config = self.Config
        
        device = Config.device

        observation_dim = self.normalizer.observation_dim
        
        dones = [0 for _ in range(num_episodes)]
        episode_rewards = [np.zeros(Config.n_agents) for _ in range(num_episodes)]
        if Config.env_type == "smac" or Config.env_type == "smac_v2":
            episode_wins = np.zeros(num_episodes)
        
        time_to_samples = []
        
        time_to_step_smac = []
        

        # TODO: reward conditioning dependent on the number of agents it seems
        returns = to_device(Config.test_ret * torch.ones(num_episodes, 1, Config.n_agents), device)
                
        env_ts = to_device(torch.arange(Config.horizon + Config.history_horizon) - Config.history_horizon, device,)
        env_ts = einops.repeat(env_ts, "t -> b t", b=num_episodes)

        t = 0
        
        obs_list = [env.reset()[None] for env in self.env_list[:num_episodes]]
        obs = np.concatenate(obs_list, axis=0)
        if Config.blind_own_type:
            obs[:, :, -3:] = -1
        recorded_obs = [deepcopy(obs[:, None])]
        
        if Config.use_hetgat_attention: 
            ntypes_list = []
            graphs_list = []
            etypes_list = []
        
            for env in self.env_list[:num_episodes]: 
                ntypes, graphs, etypes = env.get_graph_info(not self.decentralized_execution)
                ntypes_list.append(ntypes[None])
                graphs_list.append(graphs[None])
                etypes_list.append(etypes[None])
            
            ntypes = np.concatenate(ntypes_list, axis=0)
            graphs = np.concatenate(graphs_list, axis=0)
            etypes = np.concatenate(etypes_list, axis=0)
            
        
        if Config.history_horizon > 0:
            print(f"\nUsing history length of {Config.history_horizon}\n")
        else:
            print("\nDo NOT use history conditioning\n")

        obs_queue = deque(maxlen=Config.history_horizon + 1)
        
        if Config.use_hetgat_attention: 
            graphs_queue = deque(maxlen=Config.history_horizon + 1)
            etypes_queue = deque(maxlen=Config.history_horizon + 1)
            ntypes_queue = deque(maxlen=Config.history_horizon + 1)
    
        if Config.use_zero_padding:
            obs_queue.extend([np.zeros_like(obs) for _ in range(Config.history_horizon)])    
        else:
            normed_obs = self.normalizer.normalize(obs, "observations")
            obs_queue.extend([normed_obs for _ in range(Config.history_horizon)])
            
            if Config.use_hetgat_attention: 
                ntypes_queue.extend([ntypes for _ in range(Config.history_horizon)])
                graphs_queue.extend([graphs for _ in range(Config.history_horizon)])
                etypes_queue.extend([etypes for _ in range(Config.history_horizon)])
            
        while sum(dones) < num_episodes:
            
            obs = self.normalizer.normalize(obs, "observations")
            
            obs_queue.append(obs)
            if Config.use_hetgat_attention: 
                ntypes_queue.append(ntypes)
                graphs_queue.append(graphs)
                etypes_queue.append(etypes)
            
            obs = np.stack(list(obs_queue), axis=1)
            if Config.use_hetgat_attention: 
                ntypes = np.stack(list(ntypes_queue), axis=1)
                graphs = np.stack(list(graphs_queue), axis=1)    
                etypes = np.stack(list(etypes_queue), axis=1)



            
            if Config.use_hetgat_attention: 
                samples, time_to_sample = self._generate_samples(obs, 
                                                 returns, 
                                                 env_ts, 
                                                 ntypes, 
                                                 graphs,
                                                 etypes
                                                )

            else: 
                samples, time_to_sample = self._generate_samples(obs, 
                                                 returns, 
                                                 env_ts, 
                                                 )

            time_to_samples.append(time_to_sample)
            
            obs_comb = torch.cat([samples[:, 0, :, :], samples[:, 1, :, :]], dim=-1)
            obs_comb = obs_comb.reshape(-1, Config.n_agents, 2 * observation_dim) # TODO: dependent on number of agents and observation dim 
            
            if Config.share_inv or Config.joint_inv or Config.het_inv:
                
                if Config.joint_inv:
                    actions = self.trainer.ema_model.inv_model(obs_comb.reshape(obs_comb.shape[0], -1)).reshape(obs_comb.shape[0], obs_comb.shape[1], -1)
                
                elif Config.share_inv: # shared inverse model 
                    actions = self.trainer.ema_model.inv_model(obs_comb)
                
                elif Config.het_inv: # heterogeneous inverse model 
                    
                    ntypes_t = torch.tensor(ntypes[:, -1, :], device=Config.device)
                    ntypes_t = rearrange(ntypes_t, "b a -> (b a)")
                    out = rearrange(obs_comb, "(bt) a f -> (bt a) f")
                    
                    # heterogenous set of linear layers 
                    for layer_idx, layer in enumerate(self.trainer.ema_model.inv_model): 
                        out = layer(out, ntypes_t)            
                        if layer_idx < len(self.trainer.ema_model.inv_model) - 1: # pass through activation all but last layer 
                            out = self.trainer.ema_model.inv_model_activation(out)
                    actions = rearrange(out, "(b a) f -> b a f", b=obs_comb.shape[0], a=obs_comb.shape[1])
                
                
            else: # individual inverse models
                actions = torch.stack([self.trainer.ema_model.inv_model[i](obs_comb[:, i])
                                       for i in range(Config.n_agents)], dim=1,)

            samples = to_np(samples)
            actions = to_np(actions)

            if self.discrete_action:
                legal_action = np.stack([env.get_legal_actions() for env in self.env_list], axis=0)
                actions[np.where(legal_action.astype(int) == 0)] = -np.inf
                actions = np.argmax(actions, axis=-1)
            
            else:
                actions = self.normalizer.unnormalize(actions, "actions")

            if t == 0:
                normed_observations = samples[:, :, :, :]
                observations = self.normalizer.unnormalize(normed_observations, "observations")
                savepath = os.path.join("images", "sample-planned.png")
                self.trainer.renderer.composite(savepath, observations)

            obs_list = [] 
            if Config.use_hetgat_attention:
                ntypes_list = []
                graphs_list = []
                etypes_list = []
            
            for i in range(num_episodes):
                
                if dones[i] == 1:
                    obs_list.append(obs[i, 0][None])
                    if Config.use_hetgat_attention:
                        ntypes_list.append(ntypes[i, 0][None])
                        graphs_list.append(graphs[i, 0][None])
                        etypes_list.append(etypes[i, 0][None])
                
                else:
                    start_time_step_smac = time.time()

                    this_obs, this_reward, this_done, this_info = self.env_list[i].step(actions[i])                    
                    
                    time_to_step_smac.append(time.time() - start_time_step_smac)
                    
                    
                    
                    
                    
                    
                    obs_list.append(this_obs[None])
                                        
                    this_ntypes, this_graphs, this_etypes = self.env_list[i].get_graph_info()
                    if Config.use_hetgat_attention:
                        ntypes_list.append(this_ntypes[None])
                        graphs_list.append(this_graphs[None])
                        etypes_list.append(this_etypes[None])

                    if Config.use_return_to_go:
                        returns[i] = self._update_return_to_go(returns[i], this_reward)

                    if this_done.all() or t >= Config.max_path_length - 1:
                        dones[i] = 1
                        episode_rewards[i] += this_reward
                        
                        if "battle_won" in this_info.keys():
                            episode_wins[i] = this_info["battle_won"]
                            logger.print(f"Episode ({i}): battle won {episode_wins[i]}", color="green",)

                        logger.print(f"Episode ({i}): {episode_rewards[i]}", color="green")

                    else:
                        episode_rewards[i] += this_reward

            obs = np.concatenate(obs_list, axis=0)
            recorded_obs.append(deepcopy(obs[:, None]))
            if Config.use_hetgat_attention:
                graphs = np.concatenate(graphs_list, axis=0)
                etypes = np.concatenate(etypes_list, axis=0)
                ntypes = np.concatenate(ntypes_list, axis=0)
                        
            t += 1            
            env_ts = env_ts + 1
            
            # self.debug_print(f"episode step t={t}")
            # self.debug_print(f'time taken: {time.time() - start_time:.2f}s')
        
        recorded_obs = np.concatenate(recorded_obs, axis=1)
        episode_rewards = np.array(episode_rewards)

        avg_episode_time_to_sample = np.mean(time_to_samples)
        std_episode_time_to_sample = np.std(time_to_samples)
        
        avg_time_to_step_smac = np.mean(time_to_step_smac)
        std_time_to_step_smac = np.std(time_to_step_smac)
        

        if Config.env_type == "smac" or Config.env_type == "smac_v2":
            return recorded_obs, episode_rewards, episode_wins, avg_episode_time_to_sample, std_episode_time_to_sample, avg_time_to_step_smac, std_time_to_step_smac
        
        
        
        else:
            return recorded_obs, episode_rewards
        
        
    def _init(self, 
              eval_experiment_name: str , 
              log_dir: str,
              eval_env: str, 
              eval_env_subteam: Optional[str] = '',
              condition_guidance_w: Optional[float] = None, 
              prob_obs_enemy=1.0, 
              decentralized_execution=False,
              decentralized_attention=False,
              **kwargs
              ):
        
        """ Initialize the evaluator.

        log_dir: str - the directory where the trained model is saved.
        
        condition_guidance_w: float - the condition guidance weight to overwrite the trained model's condition guidance weight.
        **kwargs: dict - additional arguments to overwrite the trained model's arguments.
        
        
        Note
        decentralized_execution: use for running mad decentralized (conditioning only on )
        decentralized_attention: used for running M
        
        """
        
        assert self.initialized is False, "Evaluator can only be initialized once."

        self.eval_experiment_name = eval_experiment_name
        self.log_dir = log_dir
        self.eval_env = eval_env
        self.eval_env_subteam = eval_env_subteam 
        self.prob_obs_enemy = prob_obs_enemy
        self.decentralized_execution = decentralized_execution
        # self.decentralized_attention = 
    
        with open(os.path.join(log_dir, "parameters.pkl"), "rb") as f:
            params = pickle.load(f)

        #  builds config from the log directory (logs from trained models)
        Config = build_config_from_dict(params["Config"])
        
        # TODO: this loads the config of the trained model/experiment
        # everything is the same across compositions, except for the path length and the number of agents which the 
        # evaluation code uses
        # need to add way to specify that of the eval task 

        self.Config = Config = build_config_from_dict(kwargs, Config)

        self.Config.joint_inv = getattr(Config, "joint_inv", False)
        self.Config.share_inv = getattr(Config, "share_inv", False)
        self.Config.het_inv = getattr(Config, "het_inv", False)
        
        self.Config.use_return_to_go = getattr(Config, "use_return_to_go", False)
        self.Config.use_ddim_sample = getattr(Config, "use_ddim_sample", False)
        self.Config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.Config.blind_own_type = getattr(Config, "blind_own_type", False)
        
        logger.configure(log_dir)
        torch.backends.cudnn.benchmark = True

    
        # TODO: need to be able to load configs to run the model that was trained (source model/source env) but on 
        # a different environment (target env)
    
        with open(os.path.join(log_dir, "model_config.pkl"), "rb") as f:
            model_config = pickle.load(f)

        with open(os.path.join(log_dir, "diffusion_config.pkl"), "rb") as f:
            diffusion_config = pickle.load(f)

        with open(os.path.join(log_dir, "trainer_config.pkl"), "rb") as f:
            trainer_config = pickle.load(f)
        
        with open(os.path.join(log_dir, "dataset_config.pkl"), "rb") as f:
            dataset_config = pickle.load(f)
        
        with open(os.path.join(log_dir, "render_config.pkl"), "rb") as f:
            render_config = pickle.load(f)

        self.rewrite_cgw = False
        if condition_guidance_w is not None:
            print(f"Set condition guidance weight to {condition_guidance_w}")
            diffusion_config._dict["condition_guidance_w"] = condition_guidance_w
            self.rewrite_cgw = True

        # TODO: dataset of the trained model is needed? or the evaluation environment ? 
        # error here because of map name? 
        dataset = dataset_config()
        self.normalizer = dataset.normalizer
        self.mask_generator = dataset.mask_generator
        
        print(self.Config)
        
        del dataset
        gc.collect()

        renderer = render_config()
        model = model_config()
        diffusion = diffusion_config(model)
        
        self.trainer = trainer_config(diffusion, None, renderer)

        if Config.use_ddim_sample:
            print(f"\n Use DDIM Sampler of {Config.n_ddim_steps} Step(s) \n")
            self.trainer.model.set_ddim_scheduler(Config.n_ddim_steps)
            self.trainer.ema_model.set_ddim_scheduler(Config.n_ddim_steps)

        self.discrete_action = False        
        if Config.env_type == "smac" or Config.env_type == "smac_v2":
            self.discrete_action = True

        """ Load Environment """
        env_mod_name = {
            "d4rl": "diffuser.datasets.d4rl",
            "mahalfcheetah": "diffuser.datasets.mahalfcheetah",
            "mamujoco": "diffuser.datasets.mamujoco",
            "mpe": "diffuser.datasets.mpe",
            "smac": "diffuser.datasets.smac_env",
            "smac_v2": "diffuser.datasets.smacv2_env",
        }[Config.env_type]        
        
        env_mod = importlib.import_module(env_mod_name)

        Config.num_envs = getattr(Config, "num_envs", Config.num_eval)        
        self.env_list = [env_mod.load_environment_evaluator(self.eval_env, self.eval_env_subteam, self.prob_obs_enemy) 
                         for _ in range(Config.num_envs)]
    

        self.initialized = True


    def run(self):
        self.parent_remote.close()
        
        if not self.verbose:
            sys.stdout = open(os.devnull, "w")
        
        try:
            while True:
                try:
                    cmd, data = self.queue.get()
                
                except EOFError:
                    self.p.close()
                    break

                if cmd == "init":
                    self._init(**data)
                    
                elif cmd == "evaluate":
                    metrics = self._evaluate(**data)
                    
                    # self.p.send(metrics)
                    # self.queue.put(["eval_results", metrics])
                
                elif cmd == "close":
                    self.p.send("closed")
                    self.p.close()
                    # self.queue.shutdown()
                    break
                else:
                    self.p.close()
                    raise NotImplementedError(f"Unknown command {cmd}")

                time.sleep(1)

        except KeyboardInterrupt:
            self.p.close()

class MADEvaluator:
    def __init__(self, 
                 tb_logger=None,
                 **kwargs):
        
        multiprocessing.set_start_method("spawn", force=True)
        
        self.parent_remote, self.child_remote = Pipe()
        self.queue = multiprocessing.Queue()
        
        self._worker_process = MADEvaluatorWorker(parent_remote=self.parent_remote, 
                                                  child_remote=self.child_remote, 
                                                  queue=self.queue, **kwargs,
                                                  )
        
        self.tb_logger = tb_logger
       
        self._worker_process.start() 
        self.child_remote.close()

    def init(self, **kwargs):
        self.queue.put(["init", kwargs])

    def evaluate(self, **kwargs):
        print("Evaluating ...")
        self.queue.put(["evaluate", kwargs])
        
        # tb-logger implementation not working with multiprocessing bc would need to wait for the process to finish to write to tb 
        # self.waiting_eval = True    
        # metrics = self.parent_remote.recv()
        # for k, v in metrics.items():
        #     if isinstance(v, np.ndarray):
        #         v = np.mean(v)
        #     self.tb_logger.add_scalar(k, v, kwargs['load_step'])
        # self.waiting_eval = False
    
    def __del__(self):
        try:
            self.queue.put(["close", None])
            # mp may be deleted so it may raise AttributeError
            self.parent_remote.recv()
            self._worker_process.join()
        except (BrokenPipeError, EOFError, AttributeError, FileNotFoundError):
            pass
        # ensure the subproc is terminated
        self._worker_process.terminate()


