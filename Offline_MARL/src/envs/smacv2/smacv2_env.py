import os

from typing import Any, Dict, List, Optional
from itertools import product

import gym

import gym.spaces

import numpy as np
import pickle

import dgl 

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

    def __init__(self, scenario, map_name = None, seed = None, add_agent_ids_to_obs: bool = True, subteam: Optional[str] = '', prob_obs_enemy=1.0):
        
        dist_scenario = f"{scenario}_{subteam}" if subteam else scenario
        
        distribution_config = DISTRIBUTION_CONFIGS[dist_scenario]
        
        self.environment_label = f"smac_v2/{scenario}"
        self._environment = StarCraftCapabilityEnvWrapper(capability_config=distribution_config,
                                                          map_name=MAP_NAMES[scenario],
                                                          debug=False,
                                                          conic_fov=False,
                                                          obs_own_pos=True,
                                                          use_unit_ranges=True,
                                                          min_attack_range=2,
                                                          prob_obs_enemy=prob_obs_enemy,
                                                          )

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
        return env_info
    
    def get_graph_info(self, complete_graph=True): 
        """Gets DGL graph information for the episode. Assuming a complete graph.
  
        return ntypes: np.array of node types for each agent in the environment
        return graphs: np.array of dgl graphs
        return etypes: np.array of edge types for each agent in the environment
        """

        ntypes = self.get_agent_types()        
        
        if complete_graph: 
            # Create a complete graph for each step in the episode
            adj_matrix = np.ones((self.num_agents, self.num_agents), dtype=int)
            np.fill_diagonal(adj_matrix, 0)
        else: 
            adj_matrix = self._environment.get_visibility_matrix()[: , :self.num_agents].astype('int')
    
        src, dst = np.nonzero(adj_matrix)    
        graph = dgl.graph((src, dst), num_nodes=self.num_agents)
        graphs = np.array([graph], dtype=object).flatten()

        # retrieve all (class -> class) edge type relations in graph according the class of each agent        
        relations_dict = {}
        for i, relation in enumerate(product([0, 1, 2], [0, 1, 2])): # all SMAC envs have maximum of 3 classes
            relations_dict[relation] = i
        
        graph_edges = graph.edges() # retrieves all of the edges in the graph; src, dst agent index
        
        # gets the edge type label for each edge in the graph between two agents
        etypes = np.array([relations_dict[(int(ntypes[graph_edges[0][i]]), int(ntypes[graph_edges[1][i]]))] 
                           for i in range(graph_edges[0].size(dim=0))]).flatten()
        
        
        if (not complete_graph): # non-complete etypes tensors padded for reshaping
            num_complete_graph_edges = self.n_agents*(self.n_agents - 1)
            etypes = np.pad(etypes, (0, num_complete_graph_edges - len(etypes)), 'constant', 
                            constant_values=-1)
        
        return ntypes, graphs, etypes
    
    def get_agent_types(self): 
        """ Return agent types keeping the type indexes consistent across configurations. 
        Using indexing definitions as in SMACv2 for all class scenarios. 
        
        Terran
        -------------------
        Marine (type 0) 
        Marauder (type 1) 
        Medivac (type 2)
        
        Zerg
        -------------------
        Zergling (type 0)
        Baneling (type 1)
        Hydralisk (type 2)
        
        Protoss
        -------------------
        Stalker (type 0)
        Zealot (type 1)
        Colossus (type 2)
        """
        
        n_agents = self._environment.n_agents
        agent_classes = np.zeros(n_agents)
        
        # type smacv1 (int) -> unit (str)
        unit_type_map = {v: k for k, v in self._environment.ally_unit_map.items()}
        
        # unit (str) -> algorithm class (int)
        class_map = {'zergling': 0,
                     'baneling': 1,
                     'hydralisk': 2, 
                     
                     'marine': 0, 
                     'marauder': 1, 
                     'medivac': 2, 
                     
                     'stalker': 0, 
                     'zealot': 1, 
                     'colossus': 2, 
                     }
        
        for agent_idx, agent_unit_type in enumerate(self._environment._unit_types[:n_agents]):
            agent_classes[agent_idx] = class_map[unit_type_map[agent_unit_type]]
        
        return agent_classes
    
    

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




def load_environment_evaluator(env_name, eval_name_subteam, prob_obs_enemy=1.0):
    
    if eval_name_subteam: 
        print(f'Initializing SMACv2 evaluation environment with subteam: {eval_name_subteam}')
        
    env = SMACv2(env_name, subteam=eval_name_subteam, prob_obs_enemy=prob_obs_enemy)
    
    return env


def sequence_dataset(env, preprocess_fn):

    dataset_path = os.path.join(os.environ.get('DATASET_DIR_PREFIX'), 
                                "smac_v2", 
                                env.metadata["name"], 
                                env.metadata["data_split"],
                                )
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError("Dataset directory not found: {}".format(dataset_path))
    
    states = np.load(os.path.join(dataset_path, "states.npy"))
    observations = np.load(os.path.join(dataset_path, "obs.npy"))
    legal_actions = np.load(os.path.join(dataset_path, "legals.npy"))
    rewards = np.load(os.path.join(dataset_path, "rewards.npy"))
    actions = np.load(os.path.join(dataset_path, "actions.npy"))
    path_lengths = np.load(os.path.join(dataset_path, "path_lengths.npy"))

    # pickle load dictionary infos for scenarios
    with open(os.path.join(dataset_path, f"{env.metadata['name']}_info.pkl"), "rb") as f:
        info = pickle.load(f)
    
    if env.debugging: 
        for _ in range(10): 
            print('\n !!!! NOT LOADING IN FULL DATASET DUE TO DEBUGGING !!!!\n') 
            print('\n !!!! NOT LOADING IN FULL DATASET DUE TO DEBUGGING !!!!\n') 
            print('\n !!!! NOT LOADING IN FULL DATASET DUE TO DEBUGGING !!!!\n') 

        path_lengths = path_lengths[:128]

    
    start = 0
    for path_length in path_lengths:
        end = start + path_length
        
        episode_data = {}
        
        episode_states = states[start:end]
        
        episode_data["observations"] = observations[start:end]
        episode_data["legal_actions"] = legal_actions[start:end]
        episode_data["rewards"] = rewards[start:end]
        episode_data["actions"] = actions[start:end] 
        episode_data["terminals"] = np.zeros((path_length, observations.shape[1]), dtype=bool)
        episode_data["terminals"][-1] = True
        
        episode_data["states"] = states[start:end]

        if env.use_hetgat_attention:
            ntypes, graphs, etypes = get_graph_info(episode_states, env.num_agents, info['state_features'])
        
            episode_data["ntypes"] = ntypes
            episode_data["graphs"] = graphs
            episode_data["etypes"] = etypes
        
        yield episode_data
        
        start = end 


def sequence_multi_task_dataset(env, preprocess_fn):
    raise NotImplementedError


def get_episode_win_rates(self, episode_states, num_agents,  info): 
    """
    Get the win rates for each agent in the episode.
    """
    ep_length = episode_states.shape[0]
    for state in episode_states:
        ally_healths = []
        enemy_healths = [] 
        
        for n in range(num_agents):    
            ally_health = state[np.argwhere(np.array(info) == f'ally_health_{n}')[0,0]]
            ally_healths.append(ally_health)
            
            enemy_health = state[np.argwhere(np.array(info) == f'enemy_health_{n}')[0,0]]
            enemy_healths.append(enemy_health) 


def get_graph_info(episode_states, num_agents, info, complete_graph=True): 
    
    """
    Gets DGL graph information for the episode. Assuming a complete graph throughout the episode.
    
    TODO: for now returning each data per timestep, but this can be changed to return 
    a copy for each agent in the environment, not sure if this neccessary
    
    
    return ntypes: np.array of node types for each agent in the environment
    return graphs: np.array of dgl graphs
    return etypes: np.array of edge types for each agent in the environment
    
    """
    ep_length = episode_states.shape[0]
    state = episode_states[0]
    
    # Extract node type information from the first state in the episode
    ntypes = np.zeros((ep_length, num_agents))
    
    for n in range(num_agents):    
        tb1 = state[np.argwhere(np.array(info) == f'ally_unit_type_{n}_bit_0')[0,0]]
        tb2 = state[np.argwhere(np.array(info) == f'ally_unit_type_{n}_bit_1')[0,0]]
        tb3 = state[np.argwhere(np.array(info) == f'ally_unit_type_{n}_bit_2')[0,0]]
        
        agent_class = np.argwhere(np.array([tb1, tb2, tb3]) == 1)[0, 0]
        
        ntypes[:, n] = agent_class
    
    # Create a complete graph for each step in the episode
    adj_matrix = np.ones((num_agents, num_agents), dtype=int)
    np.fill_diagonal(adj_matrix, 0)
    src, dst = np.nonzero(adj_matrix)
    
    graph = dgl.graph((src, dst))
    graphs = np.array([graph for _ in range(ep_length)], dtype=object).reshape((ep_length,))

    # retrieve all (class -> class) edge type relations in graph according the class of each agent
    agent_classes = ntypes[0] # the class of each agent in this episode
    
    relations_dict = {}
    for i, relation in enumerate(product([0, 1, 2], [0, 1, 2])): # all SMAC envs have maximum of 3 classes
        relations_dict[relation] = i
    
    graph_edges = graph.edges() # retrieves all of the edges in the graph; src, dst agent index
    
    # gets the edge type label for each edge in the graph between two agents
    etype = np.array([relations_dict[(int(agent_classes[graph_edges[0][i]]), int(agent_classes[graph_edges[1][i]]))] 
                      for i in range(graph_edges[0].size(dim=0))])

    etypes = np.concatenate([etype for _ in range(ep_length)], axis=0).reshape((ep_length, -1)) 
    
    return ntypes, graphs, etypes  
    
    