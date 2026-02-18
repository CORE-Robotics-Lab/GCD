import os

from typing import Any, Dict, List, Optional
from itertools import product

import gym

import gym.spaces

import numpy as np
import pickle

# import dgl 


from smacv2.env.starcraft2.wrapper import StarCraftCapabilityEnvWrapper

from .smacv2_dist_configs import return_distribution_config


DISTRIBUTION_CONFIGS = return_distribution_config()

MAP_NAMES = {
    "terran_5_vs_5": "10gen_terran",
    "zerg_5_vs_5": "10gen_zerg",
    "protoss_5_vs_5": "10gen_protoss",
    "terran_10_vs_10": "10gen_terran",
}


class SMACv2(gym.Env):
    """Environment wrapper SMAC."""

    metadata = {}

    def __init__(
        self, 
        scenario, 
        map_name = None, 
        seed = None, 
        add_agent_ids_to_obs: bool = True, 
        subteam: Optional[str] = '', 
        prob_obs_enemy=1.0, 
        **kwargs,
    ):
        
        dist_scenario = f"{scenario}_{subteam}" if subteam else scenario
        
        distribution_config = DISTRIBUTION_CONFIGS[dist_scenario]
        
        self.environment_label = f"smac_v2/{scenario}"
        
        
        self._environment = StarCraftCapabilityEnvWrapper(
            capability_config=distribution_config,
            map_name=MAP_NAMES[scenario],
            # debug=False,
            conic_fov=False,
            obs_own_pos=True,
            use_unit_ranges=True,
            min_attack_range=2,
            prob_obs_enemy=prob_obs_enemy,
            **kwargs
        )        
      
        self.state_features = self._environment.env.state_feature_names
        self.obs_features = self._environment.env.obs_feature_names
          
        self._agents = [f"agent_{n}" for n in range(self._environment.n_agents)]
        self.num_agents = len(self._agents)
        self.num_actions = self._environment.n_actions
        self._reset_next_step = True
        self._done = False
        self.max_episode_length = self._environment.episode_limit
        self.add_agent_ids_to_obs = add_agent_ids_to_obs


        if add_agent_ids_to_obs:
            self.one_hot_agent_ids = []
            for i in range(self.num_agents):
                agent_id = np.eye(self.num_agents)[i]
                self.one_hot_agent_ids.append(agent_id)
                
            self.one_hot_agent_ids = np.stack(self.one_hot_agent_ids, axis=0)

        self.observation_space = [gym.spaces.Box(low=-np.inf, 
                                                 high=np.inf, 
                                                 shape=(self._environment.get_obs_size() + self.num_agents 
                                                        if add_agent_ids_to_obs else self._environment.get_obs_size(), ),) 
                                  
                                  for _ in range(self.num_agents)]
        
        self.action_space = [gym.spaces.Discrete(n=self.num_actions) for _ in range(self.num_agents)]

    def reset(self):
        """Resets the env."""

        # Reset the environment
        self._environment.reset()
        self._done = False

        observation = np.array(self.environment.get_obs())
        if self.add_agent_ids_to_obs:
            observation = np.concatenate([self.one_hot_agent_ids, observation], axis=1)
        
        return observation
    
    def get_obs(self):
        observation = np.array(self.environment.get_obs())
        if self.add_agent_ids_to_obs:
            observation = np.concatenate([self.one_hot_agent_ids, observation], axis=1)
        
        return observation


    def step(self, actions: np.ndarray):
        """ Steps in env."""

        # Step the SMAC environment
        reward, self._done, self._info = self._environment.step(actions)
        reward_n = np.array([reward for _ in range(self.num_agents)])
        done_n = np.array([self._done for _ in range(self.num_agents)])

        
        # Get the next observation
        next_observation = np.array(self._environment.get_obs())
        
        if self.add_agent_ids_to_obs:
            next_observation = np.concatenate([self.one_hot_agent_ids, next_observation], axis=1)
        
        return reward, self._done, self._info

    def env_done(self) -> bool:
        """Check if env is done."""
        return self._done

    def get_legal_actions(self) -> List:
        """Get legal actions from the environment."""
        
        legal_actions = []
        for i, _ in enumerate(self._agents):
            legal_actions.append(np.array(self._environment.get_avail_agent_actions(i), dtype="float32"))
        
        return np.array(legal_actions)

    def get_stats(self) -> Optional[Dict]:
        """Return extra stats to be logged."""
        return self._environment.get_stats()

    @property
    def agents(self) -> List:
        """Agents still alive in env (not done)."""
        return self._agents

    @property
    def possible_agents(self) -> List:
        """All possible agents in env."""
        return self._agents

    @property
    def environment(self):
        """Returns the wrapped environment."""
        return self._environment

    def __getattr__(self, name: str) -> Any:
        """Expose any other attributes of the underlying environment."""
        if hasattr(self.__class__, name):
            return self.__getattribute__(name)
        else:
            return getattr(self._environment, name)
        
    def get_env_info(self):
        
        env_info = self._environment.get_env_info()
        
        env_info['obs_shape'] += self.num_agents
        
        env_info['state_features'] = self.state_features
        env_info['obs_features'] = self.obs_features
        
        return env_info
    

    
    

def load_environment(name, debugging, use_hetgat_attention, **kwargs):
    
    if type(name) is not str:
        # name is already an environment
        return name

    idx = name.find("-")
    env_name, data_split = name[:idx], name[idx + 1 :]

    env = SMACv2(env_name, **kwargs)
    
    if hasattr(env, "metadata"):
        assert isinstance(env.metadata, dict)
    else:
        env.metadata = {}
        
    env.metadata["data_split"] = data_split
    env.metadata["name"] = env_name
    
    env.metadata["global_feats"] = ["states"]
    
    if use_hetgat_attention: 
        env.metadata["global_feats"] += ["ntypes", "graphs", "etypes"]
    
    env.use_hetgat_attention = use_hetgat_attention
    env.debugging = debugging
    
    return env

