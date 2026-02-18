import os
from typing import Any, Dict, List, Optional
from itertools import product

import gym
import numpy as np
import pickle

import dgl 

from smac.env import StarCraft2Env


class SMAC(gym.Env):
    """Environment wrapper SMAC."""

    metadata = {}

    def __init__(self, map_name: str, add_agent_ids_to_obs: bool = True):
        self._environment = StarCraft2Env(map_name=map_name, obs_last_action=False)
        self._agents = [f"agent_{n}" for n in range(self._environment.n_agents)]
        self.num_agents = len(self._agents)
        self.num_actions = self._environment.n_actions
        self._done = False
        self.max_episode_length = self._environment.episode_limit
        self.add_agent_ids_to_obs = add_agent_ids_to_obs

        if add_agent_ids_to_obs:
            self.one_hot_agent_ids = []
            for i in range(self.num_agents):
                agent_id = np.eye(self.num_agents)[i]
                self.one_hot_agent_ids.append(agent_id)
            self.one_hot_agent_ids = np.stack(self.one_hot_agent_ids, axis=0)

        self.observation_space = [
            gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(
                    self._environment.get_obs_size() + self.num_agents
                    if add_agent_ids_to_obs
                    else self._environment.get_obs_size(),
                ),
            )
            for _ in range(self.num_agents)
        ]
        self.action_space = [
            gym.spaces.Discrete(n=self.num_actions) for _ in range(self.num_agents)
        ]

    def reset(self):
        """Resets the env."""

        # Reset the environment
        self._environment.reset()
        self._done = False

        observation = np.array(self._environment.get_obs())
        if self.add_agent_ids_to_obs:
            observation = np.concatenate([self.one_hot_agent_ids, observation], axis=1)
        return observation

    def step(self, actions: np.ndarray):
        """Steps in env."""

        # Step the SMAC environment
        reward, self._done, self._info = self._environment.step(actions)
        reward_n = np.array([reward for _ in range(self.num_agents)])
        done_n = np.array([self._done for _ in range(self.num_agents)])

        # Get the next observation
        next_observation = np.array(self._environment.get_obs())
        if self.add_agent_ids_to_obs:
            next_observation = np.concatenate(
                [self.one_hot_agent_ids, next_observation], axis=1
            )
        return next_observation, reward_n, done_n, self._info

    def env_done(self) -> bool:
        """Check if env is done."""
        return self._done

    def get_legal_actions(self) -> List:
        """Get legal actions from the environment."""
        legal_actions = []
        for i, _ in enumerate(self._agents):
            legal_actions.append(
                np.array(self._environment.get_avail_agent_actions(i), dtype="float32")
            )
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
    
    def get_agent_types(self): 
        """ 
        return agent types keeping the type indexes consistent across configurations. using indexing SMACv2 all 
        class scenarios. 
        
        
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
        
        ally_unit_types = self._environment._unit_types[:self._environment.n_agents]
        agent_classes = np.zeros(self._environment.n_agents)
        
        if self._environment.map_type == "marines":
            for agent_idx, ally_unit_type in enumerate(ally_unit_types): 
                if ally_unit_types == self._environment.marine_id:
                    agent_classes[agent_idx] = 0
        
        elif self._environment.map_type == "stalkers_and_zealots":
            for agent_idx, ally_unit_type in enumerate(ally_unit_types): 
                if ally_unit_types == self._environment.stalker_id:
                    agent_classes[agent_idx] = 0
                elif ally_unit_types == self._environment.zealot_id:
                    agent_classes[agent_idx] = 1
            
        elif self._environment.map_type == "colossi_stalkers_zealots":
            for agent_idx, ally_unit_type in enumerate(ally_unit_types): 
                if ally_unit_types == self._environment.stalker_id:
                    agent_classes[agent_idx] = 0
                elif ally_unit_types == self._environment.zealot_id:
                    agent_classes[agent_idx] = 1
                elif ally_unit_types == self._environment.colossus_id:
                    agent_classes[agent_idx] = 2

        elif self._environment.map_type == "MMM":
            for agent_idx, ally_unit_type in enumerate(ally_unit_types):
                if ally_unit_type == self._environment.marine_id:
                    agent_classes[agent_idx] = 0
                
                elif ally_unit_type == self._environment.marauder_id:
                    agent_classes[agent_idx] = 1
                
                elif ally_unit_type == self._environment.medivac_id:
                    agent_classes[agent_idx] = 2
        
        elif self._environment.map_type == "zealots":
            for agent_idx, ally_unit_type in enumerate(ally_unit_types):
                if ally_unit_type == self._environment.zealot_id:
                    agent_classes[agent_idx] = 1
        
        elif self._environment.map_type == "hydralisks":
            for agent_idx, ally_unit_type in enumerate(ally_unit_types):
                if ally_unit_type == self._environment.hydralisk_id:
                    agent_classes[agent_idx] = 2
            
        elif self._environment.map_type == "stalkers":
            for agent_idx, ally_unit_type in enumerate(ally_unit_types):
                if ally_unit_type == self._environment.stalker_id:
                    agent_classes[agent_idx] = 0
        
        elif self._environment.map_type == "colossus":
            for agent_idx, ally_unit_type in enumerate(ally_unit_types):
                if ally_unit_type == self._environment.colossus_id:
                    agent_classes[agent_idx] = 2
            
        elif self._environment.map_type == "bane":
            for agent_idx, ally_unit_type in enumerate(ally_unit_types):
                if ally_unit_type == self._environment.zergling_id:
                    agent_classes[agent_idx] = 0
                elif ally_unit_type == self._environment.baneling_id:
                    agent_classes[agent_idx] = 1
                    
        return agent_classes


    def get_graph_info(self, complete_graph=True): 
        """
        Gets DGL graph information for the episode. Assuming a complete graph.
  
        return ntypes: np.array of node types for each agent in the environment
        return graphs: np.array of dgl graphs
        return etypes: np.array of edge types for each agent in the environment
        
        """
        ntypes = self.get_agent_types()        
                
        # Create a complete graph for each step in the episode
        adj_matrix = np.ones((self.num_agents, self.num_agents), dtype=int)
        np.fill_diagonal(adj_matrix, 0)
        src, dst = np.nonzero(adj_matrix)
        
        graph = dgl.graph((src, dst))
        graphs = np.array([graph], dtype=object).flatten()

        # retrieve all (class -> class) edge type relations in graph according the class of each agent        
        relations_dict = {}
        for i, relation in enumerate(product([0, 1, 2], [0, 1, 2])): # all SMAC envs have maximum of 3 classes
            relations_dict[relation] = i
        
        graph_edges = graph.edges() # retrieves all of the edges in the graph; src, dst agent index
        
        # gets the edge type label for each edge in the graph between two agents
        etypes = np.array([relations_dict[(int(ntypes[graph_edges[0][i]]), int(ntypes[graph_edges[1][i]]))] 
                           for i in range(graph_edges[0].size(dim=0))]).flatten()
        
        return ntypes, graphs, etypes  

def load_environment(name, **kwargs):
    if type(name) is not str:
        # name is already an environment
        return name

    idx = name.find("-")
    if idx == -1: # specifying the eval environment, not for use with dataset intitialization
        env_name = name
        data_split = None
    else: 
        env_name, data_split = name[:idx], name[idx + 1 :]
        
    env = SMAC(env_name, **kwargs)
    if hasattr(env, "metadata"):
        assert isinstance(env.metadata, dict)
    else:
        env.metadata = {}
    
    # env.metadata["data_split"] = data_split
    
    
    env.metadata["name"] = env_name
    env.metadata["data_split"] = data_split
    env.metadata["global_feats"] = ["states", "ntypes", "graphs", "etypes"]
    
    return env


def sequence_dataset(env, preprocess_fn):
    
    # dataset_path = os.path.join(os.path.dirname(__file__), "data/smac", env.metadata["name"], env.metadata["data_split"],)
    dataset_path = os.path.join(os.environ.get('DATASET_DIR_PREFIX'), "smac", env.metadata["name"], env.metadata["data_split"],)
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError("Dataset directory not found: {}".format(dataset_path))

    states = np.load(os.path.join(dataset_path, "states.npy"))
    
    observations = np.load(os.path.join(dataset_path, "obs.npy"))
    legal_actions = np.load(os.path.join(dataset_path, "legals.npy"))
    rewards = np.load(os.path.join(dataset_path, "rewards.npy"))
    actions = np.load(os.path.join(dataset_path, "actions.npy"))
    path_lengths = np.load(os.path.join(dataset_path, "path_lengths.npy"))

    # do only once since we're passing the env object + the types are not changing 
    env.reset() # need to call reset to initialize the unit types
    agent_types = env.get_agent_types()
   
    
    start = 0
    for path_length in path_lengths:
        end = start + path_length
        episode_data = {}
        
        episode_data["observations"] = observations[start:end]
        episode_data["legal_actions"] = legal_actions[start:end]
        episode_data["rewards"] = rewards[start:end]
        episode_data["actions"] = actions[start:end]
        episode_data["terminals"] = np.zeros((path_length, observations.shape[1]), dtype=bool)
        episode_data["terminals"][-1] = True
        
        ntypes, graphs, etypes = get_graph_info(agent_types, 
                                                num_agents=env.num_agents,
                                                ep_length=path_length
                                                )
        episode_data["ntypes"] = ntypes
        episode_data["graphs"] = graphs
        episode_data["etypes"] = etypes
        
        yield episode_data
        
        start = end



def get_graph_info(agent_types, num_agents, ep_length, complete_graph=True): 
    """
    Gets DGL graph information for the episode. Assuming a complete graph throughout the episode.
    
    TODO: for now returning each data per timestep, but this can be changed to return 
    a copy for each agent in the environment, not sure if this neccessary
    
    return ntypes: np.array of node types for each agent in the environment
    return graphs: np.array of dgl graphs
    return etypes: np.array of edge types for each agent in the environment
    
    """
    
    ntypes = np.stack([agent_types for _ in range(ep_length)], axis=0)
    
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


