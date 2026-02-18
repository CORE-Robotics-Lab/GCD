
import os

from itertools import product

import numpy as np
import pickle
import gym
import dgl 
import argparse

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import json 
import pickle

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





class DataSetProcessor:
    def __init__(self, env_name, map_name, quality):
    
        self.env_name = env_name
        self.map_name = map_name
        self.quality = quality
        
        self.dataset_episodes = [] # list with a dictionary for each episode in the dataset
        
        self.init_dataset(self.env_name, self.map_name, self.quality)

        self.num_episodes = len(self.dataset_episodes)

        # self.process_reward_distribution(self.dataset_episodes, team_comp=self.map_name)
        # self.process_dataset_stats(self.dataset_episodes, team_comp=self.map_name)
        # self.split_team_compositions(self.dataset_episodes) 
        
        # self.process_dataset_stats_episodes(self.dataset_episodes,)
        
    
        # self.process_dataset_reward_filtered(self.dataset_episodes, reward_threshold=9.218)
        
        self.process_dataset_reward_filtered_team_comps(self.dataset_episodes, reward_threshold=9.0, team_comps=['2z_3b', '3z_2b', '1z_4b', '2z_2b_1h'])
        
        # self.process_win_rate(self.dataset_episodes)
        
        
        
    
                  
    def init_dataset(self, env_name, map_name, quality): 
        
        dataset_path = os.path.join(os.environ.get('DATASET_DIR_PREFIX'), f"{env_name}", f'{map_name}', f'{quality}',)
    
        if not os.path.exists(dataset_path):
            raise FileNotFoundError("Dataset directory not found: {}".format(dataset_path))
        
        states = np.load(os.path.join(dataset_path, "states.npy"))
        observations = np.load(os.path.join(dataset_path, "obs.npy"))
        legal_actions = np.load(os.path.join(dataset_path, "legals.npy"))
        rewards = np.load(os.path.join(dataset_path, "rewards.npy"))
        actions = np.load(os.path.join(dataset_path, "actions.npy"))
        path_lengths = np.load(os.path.join(dataset_path, "path_lengths.npy"))

        # pickle load dictionary infos for scenarios
        with open(os.path.join(dataset_path, f"{map_name}_info.pkl"), "rb") as f:
            info = pickle.load(f)
        
        print("loading dataset...")
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
            
            ntypes, graphs, etypes = get_graph_info(episode_states, info['env_info']['n_agents'], info['state_features'])
            
            episode_data["ntypes"] = ntypes
            episode_data["graphs"] = graphs
            episode_data["etypes"] = etypes
            
            episode_data["path_length"] = path_length 
            
            self.dataset_episodes.append(episode_data)
            
            start = end
        
        print("dataset loaded.")
    

    
    def split_team_compositions(self, episodes, max_episode_length=200, num_agents=5):
        
        team_comps_per_episode = []
                
        for ep_idx, episode in enumerate(episodes): 
            ntypes = episode["ntypes"]
            team_comp = self.team_comp_label(ntypes[0])
            team_comps_per_episode.append(team_comp)
    
        unique_team_comps = sorted(list(set(team_comps_per_episode)))    
        
        for team_comp in unique_team_comps: 
            team_comp_idxs = np.argwhere(np.array(team_comps_per_episode) == team_comp).flatten()
            team_comp_episodes = [episodes[idx] for idx in team_comp_idxs]
            
            print(f"processing team composition... : {team_comp}")
            
            self.process_reward_distribution(team_comp_episodes, team_comp=team_comp)
            self.process_dataset_stats(team_comp_episodes, team_comp=team_comp)                      
            self.split_dataset(team_comp_episodes, team_comp=team_comp) 
    
            print('finished\n')
            
    def split_dataset(self, episodes, team_comp): 
        """     
        e.g. team_comp = '1ma_1mr_1md' 
        smac_v2/terran_5_vs_5/split/1ma_1mr_1md/ 
        
        states, actions, rewards, legal_actions, observations shape: (num_samples, **)
        path_legths shape: (num_episodes, )
        """
        
        dataset_path = os.path.join(os.environ.get('DATASET_DIR_PREFIX'), 
                                    f"{self.env_name}",
                                    f'{self.map_name}', 
                                    'split', 
                                    f'{team_comp}')
        
        # make sure it exists
        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path)
            
        all_states = []
        all_actions = []
        all_rewards = []
        all_legal_actions = []
        all_observations = []
        
        all_path_lengths = []
        
        for episode in episodes:
            all_states.append(episode["states"])
            all_actions.append(episode["actions"])
            all_rewards.append(episode["rewards"])
            all_legal_actions.append(episode["legal_actions"])
            all_observations.append(episode["observations"])
            
            all_path_lengths.append(episode["path_length"])
        
        states = np.concatenate(all_states, axis=0)
        actions = np.concatenate(all_actions, axis=0)
        rewards = np.concatenate(all_rewards, axis=0)
        legal_actions = np.concatenate(all_legal_actions, axis=0)
        observations = np.concatenate(all_observations, axis=0)
        
        path_lengths = np.array(all_path_lengths)
        
        assert states.shape[0] == actions.shape[0] == rewards.shape[0] == legal_actions.shape[0] == observations.shape[0] == np.sum(path_lengths)
        print(f'saving dataset split {team_comp} to {dataset_path}')
        
        np.save(f"{dataset_path}/states.npy", states,)
        np.save(f"{dataset_path}/actions.npy", actions,)
        np.save(f"{dataset_path}/rewards.npy", rewards,)
        np.save(f"{dataset_path}/legals.npy", legal_actions,)
        np.save(f"{dataset_path}/obs.npy", observations,)
        np.save(f"{dataset_path}/path_lengths.npy", path_lengths,)



    def process_dataset_reward_filtered(self, episodes, max_episode_length=200, num_agents=5, reward_threshold=10.0): 
        """     
        e.g. team_comp = '1ma_1mr_1md' 
        smac_v2/terran_5_vs_5/split/1ma_1mr_1md/ 
        
        states, actions, rewards, legal_actions, observations shape: (num_samples, **)
        path_legths shape: (num_episodes, )
        """
        
        dataset_path = os.path.join(os.environ.get('DATASET_DIR_PREFIX'), 
                                    f"{self.env_name}",
                                    f'{self.map_name}',
                                    f'RewardThreshold{str(reward_threshold)}'
                                    )
        
        # make sure it exists
        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path)
            
        all_states = []
        all_actions = []
        all_rewards = []
        all_legal_actions = []
        all_observations = []
        
        all_path_lengths = []
        
        for episode in episodes:
            rewards = episode["rewards"]
            
            if rewards[:, 0].sum() >= reward_threshold:
            
                all_states.append(episode["states"])
                all_actions.append(episode["actions"])
                all_rewards.append(rewards)
                all_legal_actions.append(episode["legal_actions"])
                all_observations.append(episode["observations"])
                all_path_lengths.append(episode["path_length"])
        
        states = np.concatenate(all_states, axis=0)
        actions = np.concatenate(all_actions, axis=0)
        rewards = np.concatenate(all_rewards, axis=0)
        legal_actions = np.concatenate(all_legal_actions, axis=0)
        observations = np.concatenate(all_observations, axis=0)
        
        path_lengths = np.array(all_path_lengths)
        
        assert states.shape[0] == actions.shape[0] == rewards.shape[0] == legal_actions.shape[0] == observations.shape[0] == np.sum(path_lengths)
        
        print(f'saving multi-task dataset to {dataset_path}')
        
        np.save(f"{dataset_path}/states.npy", states,)
        np.save(f"{dataset_path}/actions.npy", actions,)
        np.save(f"{dataset_path}/rewards.npy", rewards,)
        np.save(f"{dataset_path}/legals.npy", legal_actions,)
        np.save(f"{dataset_path}/obs.npy", observations,)
        np.save(f"{dataset_path}/path_lengths.npy", path_lengths,)







    def process_dataset_reward_filtered_team_comps(self, episodes, max_episode_length=200, num_agents=5, reward_threshold=10.0, team_comps=['1ma_1mr_1md']): 
        """     
        e.g. team_comp = '1ma_1mr_1md' 
        smac_v2/terran_5_vs_5/split/1ma_1mr_1md/ 
        
        states, actions, rewards, legal_actions, observations shape: (num_samples, **)
        path_legths shape: (num_episodes, )
        """
        
        dataset_path = os.path.join(os.environ.get('DATASET_DIR_PREFIX'), 
                                    f"{self.env_name}",
                                    f'{self.map_name}',
                                    f'RewardThreshold{int(reward_threshold)}', 
                                    f'train_split'
                                    )
        
        # make sure it exists
        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path)
            
        all_states = []
        all_actions = []
        all_rewards = []
        all_legal_actions = []
        all_observations = []
        
        all_path_lengths = []
        
        for episode in episodes:
            rewards = episode["rewards"]
            
            ntypes = episode["ntypes"]
            team_comp = self.team_comp_label(ntypes[0])
            
            
            if rewards[:, 0].sum() >= reward_threshold and team_comp in team_comps:
            
                all_states.append(episode["states"])
                all_actions.append(episode["actions"])
                
                all_rewards.append(rewards)
                
                all_legal_actions.append(episode["legal_actions"])
                all_observations.append(episode["observations"])
                all_path_lengths.append(episode["path_length"])
        
        states = np.concatenate(all_states, axis=0)
        actions = np.concatenate(all_actions, axis=0)
        rewards = np.concatenate(all_rewards, axis=0)
        legal_actions = np.concatenate(all_legal_actions, axis=0)
        observations = np.concatenate(all_observations, axis=0)
        
        path_lengths = np.array(all_path_lengths)
        
        assert states.shape[0] == actions.shape[0] == rewards.shape[0] == legal_actions.shape[0] == observations.shape[0] == np.sum(path_lengths)
        
        print(f'saving multi-task dataset to {dataset_path}')
        
        np.save(f"{dataset_path}/states.npy", states,)
        np.save(f"{dataset_path}/actions.npy", actions,)
        np.save(f"{dataset_path}/rewards.npy", rewards,)
        np.save(f"{dataset_path}/legals.npy", legal_actions,)
        np.save(f"{dataset_path}/obs.npy", observations,)
        np.save(f"{dataset_path}/path_lengths.npy", path_lengths,)
        
        with open(f"{dataset_path}/team_comps.txt", "w") as f:
            for team_comp in team_comps:
                f.write(f"{team_comp}\n")
            






    def team_comp_label(self, ntypes): 
        """
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
        team_comp_label = ''  
        if self.map_name == 'terran_5_vs_5': 
            
            num_marine = np.sum(ntypes == 0)
            num_marauder = np.sum(ntypes == 1)
            num_medivac = np.sum(ntypes == 2)
            
            if num_marine != 0:
                team_comp_label += f'{num_marine}ma'

            if num_marine != 0 and num_marauder != 0:
                team_comp_label += f'_{num_marauder}mr'
            elif num_marauder != 0:
                team_comp_label += f'{num_marauder}mr'
            
            if (num_marine != 0 or num_marauder != 0) and num_medivac != 0:
                team_comp_label += f'_{num_medivac}md'
                
            elif num_medivac != 0:
                team_comp_label += f'{num_medivac}md'
                
                
        if self.map_name == 'zerg_5_vs_5': 
            
            num_zergling = np.sum(ntypes == 0)
            num_baneling = np.sum(ntypes == 1)
            num_hydralisk = np.sum(ntypes == 2)
            
            if num_zergling != 0:
                team_comp_label += f'{num_zergling}z'

            if num_zergling != 0 and num_baneling != 0:
                team_comp_label += f'_{num_baneling}b'
            elif num_baneling != 0:
                team_comp_label += f'{num_baneling}b'
            
            if (num_zergling != 0 or num_baneling != 0) and num_hydralisk != 0:
                team_comp_label += f'_{num_hydralisk}h'
                
            elif num_hydralisk != 0:
                team_comp_label += f'{num_hydralisk}h'
            

        return team_comp_label


    def process_dataset_stats(self, episodes, team_comp='terran_5_vs_5'):
        stats = {}
        
        stats['num_episodes'] = str(len(episodes))
        stats['max_episode_length'] = str(max([episode["path_length"] for episode in episodes]))
        stats['num_samples'] = str(sum([episode["path_length"] for episode in episodes]))   

        save_path = os.path.join('stats', f'{self.env_name}', f'{self.map_name}',)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            
        with open(f'{save_path}/{team_comp}.json', 'w') as f:
            json.dump(stats, f)
            
            
    def process_reward_distribution(self, episodes, max_episode_length=200, num_agents=5, team_comp='terran_5_vs_5'):
    
        num_episodes = len(episodes)
        rewards = np.zeros((num_episodes, max_episode_length, num_agents), dtype=np.float32)
        for ep_idx, episode in enumerate(episodes): 
            rewards[ep_idx, :episode["path_length"]] = episode["rewards"]
        
        episode_rewards = np.sum(rewards[:, :, 0], axis=1) 
        
        save_path = os.path.join('reward_dist', f'{self.env_name}', f'{self.map_name}',)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        with open(f"{save_path}/{team_comp}.pkl", "wb") as f:
            pickle.dump(episode_rewards, f)
        
        
    def process_dataset_stats_episodes(self, episodes, max_episode_length=200, num_agents=5, team_comp='terran_5_vs_5'):
        
        ep_team_comps = []
        ep_lengths = []
        
        
        num_episodes = len(episodes)
        rewards = np.zeros((num_episodes, max_episode_length, num_agents), dtype=np.float32)
        
        for ep_idx, episode in enumerate(episodes): 
            ntypes = episode["ntypes"]
            team_comp_label = self.team_comp_label(ntypes[0])
            ep_team_comps.append(team_comp_label)
            
            ep_lengths.append(episode["path_length"])
            
            rewards[ep_idx, :episode["path_length"]] = episode["rewards"]
            
        ep_returns = np.sum(rewards[:, :, 0], axis=1).tolist()
        
        
        save_path = os.path.join('episode_stats', f'{self.env_name}', f'{self.map_name}',)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        with open(f"{save_path}/{team_comp}_episode_stats.pkl", "wb") as f:
            pickle.dump((ep_team_comps, ep_lengths, ep_returns), f)
       
    
    
    def process_win_rate(self, episodes, max_episode_length=200, num_agents=5, team_comp='terran_5_vs_5'):
        """ TODO: """
        dataset_path = os.path.join(os.environ.get('DATASET_DIR_PREFIX'), f"{self.env_name}", f'{self.map_name}', f'{self.quality}',)
        
        # pickle load dictionary infos for scenarios
        with open(os.path.join(dataset_path, f"{self.map_name}_info.pkl"), "rb") as f:
            info = pickle.load(f)
            
        n_agents = info['env_info']['n_agents']
        state_features = info['state_features']
    
        for ep_idx, episode in enumerate(episodes):    
            ep_final_state = episode["states"][-1]
            
            ep_healths = []
            enemy_healths = []
            
            for t in range(len(episode["states"])):
                ep_final_state = episode["states"][t]
                
                ally_healths = []
                enemy_healths = [] 
                
                for agent_idx in range(n_agents): 
                    idx = np.argwhere(np.array(state_features) == f'ally_health_{agent_idx}')[0,0]
                    ally_health = ep_final_state[idx]
                    ally_healths.append(ally_health)
                    
                    idx = np.argwhere(np.array(state_features) == f'enemy_health_{agent_idx}')[0,0]
                    enemy_health = ep_final_state[idx]
                    enemy_healths.append(enemy_health)
    
            aa = 10 
            
    
    
    def process_reward_plots(self, episodes, max_episode_length=200, num_agents=5, team_comp='terran_5_vs_5'):
        
        num_episodes = len(episodes)
        rewards = np.zeros((num_episodes, max_episode_length, num_agents), dtype=np.float32)
        for ep_idx, episode in enumerate(episodes): 
            rewards[ep_idx, :episode["path_length"]] = episode["rewards"]
        episode_rewards = np.sum(rewards[:, :, 0], axis=1) 
        
        df = pd.DataFrame(episode_rewards, columns=["rewards"])
        plt.figure(figsize=(10, 6))
        sns.violinplot(data=df, y="rewards")
        
        if self.map_name == 'terran_5_vs_5':
            plt.ylim(-2, 40)
        
        plt.title(f"reward distribution - {team_comp}")
        plt.show()

        if not os.path.exists("figures"):
            os.makedirs("figures")
        plt.savefig(f"figures/reward_distribution_{team_comp}.png")
        
        

    
    
    


       

if __name__ == "__main__": 

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--env_name", type=str, default="smac")
    parser.add_argument("--map_name", type=str, default="3m")
    parser.add_argument("--quality", type=str, default="Good")
    
    args = parser.parse_args()
    
    ds_procc = DataSetProcessor(args.env_name, args.map_name, args.quality) 
    
    

