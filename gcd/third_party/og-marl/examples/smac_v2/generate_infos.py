



from __future__ import absolute_import, division, print_function

import time
from os import replace
import pickle
import numpy as np
from absl import logging
from smacv2.env import StarCraft2Env
from smacv2.env.starcraft2.wrapper import StarCraftCapabilityEnvWrapper

logging.set_verbosity(logging.DEBUG)


def main(scenario_dist_configs, map_name='10gen_terran'):


    
    for scenario_name, distribution_config in scenario_dist_configs.items():
        
        env = StarCraftCapabilityEnvWrapper(capability_config=distribution_config, 
                                            map_name=map_name,
                                            debug=False,
                                            conic_fov=False,
                                            use_unit_ranges=True,
                                            min_attack_range=2,
                                            obs_own_pos=True,
                                            fully_observable=False,
                                            )
        env.reset()
    
        obs = env.get_obs()
        state = env.get_state()
        cap = env.get_capabilities()
        
        # agent_classes = [np.argwhere(obs[i][-3:])[0][0] for i in range(env.n_agents)]
        # print(f'agent classes {agent_classes}')
        
        all_information_dict = {}
        all_information_dict['env_info']  = env.get_env_info()
        all_information_dict['state_features'] =  env.get_state_feature_names()
        all_information_dict['obs_features'] = env.get_obs_feature_names()
        
        
        with open(f'{scenario_name}_info.pkl', 'wb') as f:
            print(f'saving {scenario_name}_info.pkl')
            pickle.dump(all_information_dict, f)
            
        env.close()
        
        
        


if __name__ == "__main__":
    
    
    distribution_config_terran = {
        "n_units": 5,
        "n_enemies": 5,
        "team_gen": {
            "dist_type": "weighted_teams",
            "unit_types": ["marine", "marauder", "medivac"],
            
            "weights": [0.45, 0.45, 0.1],
            
            "observe": True,
            "exception_unit_types": ["medivac"],
        },
        
        "start_positions": {
            "dist_type": "surrounded_and_reflect",
            "p": 0.5,
            "map_x": 32,
            "map_y": 32,
        }
    }
    
    
    
    distribution_config_zerg = {
        "n_units": 5,
        "n_enemies": 5,
        
        "team_gen": {
            "dist_type": "weighted_teams",
            "unit_types": ["zergling", "baneling", "hydralisk"],
            "exception_unit_types": ["baneling"],
            "weights": [0.45, 0.1, 0.45],
            "observe": True,
        },
        
        "start_positions": {
            "dist_type": "surrounded_and_reflect",
            "p": 0.5,
            "n_enemies": 5,
            "map_x": 32,
            "map_y": 32,
        },
    }
    
    scenario_dist_configs = {}
    
    # scenario_dist_configs['terran_5_vs_5'] = distribution_config_terran.copy()
    
    
    # scenario_dist_configs['terran_10_vs_10'] = distribution_config_terran.copy()
    # scenario_dist_configs['terran_10_vs_10']['n_units'] = 10
    # scenario_dist_configs['terran_10_vs_10']['n_enemies'] = 10
    
    
    scenario_dist_configs['zerg_5_vs_5'] = distribution_config_zerg.copy()
    
    
    main(scenario_dist_configs, map_name='10gen_zerg')
