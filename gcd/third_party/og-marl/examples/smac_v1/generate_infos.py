
import numpy as np 
import pickle as pkl 
from smac.env import StarCraft2Env



def main(scenarios):


    
    for scenario_name in scenarios:
        
        env = StarCraft2Env(map_name='3m')

        env.reset()
    
        obs = env.get_obs()
        state = env.get_state()
        
        # agent_classes = [np.argwhere(obs[i][-3:])[0][0] for i in range(env.n_agents)]
        # print(f'agent classes {agent_classes}')
        
        all_information_dict = {}
        all_information_dict['env_info']  = env.get_env_info()
        all_information_dict['state_features'] =  get_state_feature_names(env)
        # all_information_dict['obs_features'] = get_obs_feature_names(env)
        
        
        info = all_information_dict['state_features']
        
        
        # Extract node type information from the first state in the episode
        ntypes = np.zeros((env.n_agents,))
        
        for n in range(env.n_agents):    
            tbs = []
            for b in range(env.unit_type_bits):
                tb = state[np.argwhere(np.array(info) == f'enemy_unit_type_{n}_bit_{b}')[0,0]]
                tbs.append(tb)
            
            agent_class = np.argwhere(np.array(tbs) == 1)[0, 0]
            
            ntypes[n] = agent_class
        
        
        ntypes_enemy = np.zeros((env.n_enemies,))
        
        for n in range(env.n_enemies):
            tbs = []
            for b in range(env.unit_type_bits):
                tb = state[np.argwhere(np.array(info) == f'enemy_unit_type_{n}_bit_{b}')[0,0]]
                tbs.append(tb)
            
            agent_class = np.argwhere(np.array(tbs) == 1)[0, 0]
            
            ntypes_enemy[n] = agent_class
    
        with open(f'{scenario_name}_info.pkl', 'wb') as f:
            print(f'saving {scenario_name}_info.pkl')
            pkl.dump(all_information_dict, f)
            
        env.close()
        

def get_agent_types(env): 
    """ return agent types keeping types consistent across configurations
    
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
    
    ally_unit_type_ids = env._unit_types[:env.n_agents]
    
    
    if env.map_type == "marines":
        env.marine_id = min_unit_type
    
    elif env.map_type == "stalkers_and_zealots":
        env.stalker_id = min_unit_type
        env.zealot_id = min_unit_type + 1
    
    elif env.map_type == "colossi_stalkers_zealots":
        env.colossus_id = min_unit_type
        env.stalker_id = min_unit_type + 1
        env.zealot_id = min_unit_type + 2
    
    elif env.map_type == "MMM":
        
        agent_classes = np.zeros(env.n_agents)
        
        for agent_idx, ally_unit_type_id in enumerate(ally_unit_type_ids):
            if ally_unit_type_id == env.marine_id:
                agent_classes[agent_idx] = 0
            
            elif ally_unit_type_id == env.marauder_id:
                agent_classes[agent_idx] = 1
            
            elif ally_unit_type_id == env.medivac_id:
                agent_classes[agent_idx] = 2
    
    elif env.map_type == "zealots":
        agent_classes = np.zeros(env.n_agents)
        for agent_idx, ally_unit_type_id in enumerate(ally_unit_type_ids):
            if ally_unit_type_id == env.zealot_id:
                agent_classes[agent_idx] = 1
        
    
    
    elif env.map_type == "hydralisks":
        env.hydralisk_id = min_unit_type
    
    elif env.map_type == "stalkers":
        env.stalker_id = min_unit_type
    
    elif env.map_type == "colossus":
        env.colossus_id = min_unit_type
    
    elif env.map_type == "bane":
        env.baneling_id = min_unit_type
        env.zergling_id = min_unit_type + 1

    
            
        
        

# function from the smacv2 codebase 
def get_state_feature_names(env):
    """Return the state feature names."""
    if env.obs_instead_of_state:
        raise NotImplementedError

    feature_names = []

    # Ally features.
    for al_id in range(env.n_agents):
        feature_names.append(f"ally_health_{al_id}")
        feature_names.append(f"ally_cooldown_{al_id}")
        feature_names.append(f"ally_relative_x_{al_id}")
        feature_names.append(f"ally_relative_y_{al_id}")

        if env.shield_bits_ally > 0:
            feature_names.append(f"ally_shield_{al_id}")


        if env.unit_type_bits > 0:
            for bit in range(env.unit_type_bits):
                feature_names.append(f"ally_unit_type_{al_id}_bit_{bit}")

    # Enemy features.
    for e_id in range(env.n_enemies):
        feature_names.append(f"enemy_health_{e_id}")
        feature_names.append(f"enemy_relative_x_{e_id}")
        feature_names.append(f"enemy_relative_y_{e_id}")

        if env.shield_bits_enemy > 0:
            feature_names.append(f"enemy_shield_{e_id}")

        if env.unit_type_bits > 0:
            for bit in range(env.unit_type_bits):
                feature_names.append(f"enemy_unit_type_{e_id}_bit_{bit}")

    if env.state_last_action:
        for al_id in range(env.n_agents):
            for action_idx in range(env.n_actions):
                feature_names.append(
                    f"ally_last_action_{al_id}_action_{action_idx}"
                )

    if env.state_timestep_number:
        feature_names.append("timestep")

    return feature_names


# TODO: fix this, don't need it rn
def get_obs_feature_names(env):
        """Return the observations feature names."""
        feature_names = []

        # Movement features.
        feature_names.extend(
            [
                "move_action_north",
                "move_action_south",
                "move_action_east",
                "move_action_west",
            ]
        )
        if env.obs_pathing_grid:
            feature_names.extend(
                [f"pathing_grid_{n}" for n in range(env.n_obs_pathing)]
            )
        if env.obs_terrain_height:
            feature_names.extend(
                [f"terrain_height_{n}" for n in range(env.n_obs_height)]
            )

        # Enemy features.
        for e_id in range(env.n_enemies):
            feature_names.extend(
                [
                    f"enemy_shootable_{e_id}",
                    f"enemy_distance_{e_id}",
                    f"enemy_relative_x_{e_id}",
                    f"enemy_relative_y_{e_id}",
                ]
            )
            if env.obs_all_health:
                feature_names.append(f"enemy_health_{e_id}")
            if env.obs_all_health and env.shield_bits_enemy > 0:
                feature_names.append(f"enemy_shield_{e_id}")
            if env.unit_type_bits > 0:
                feature_names.extend(
                    [
                        f"enemy_unit_type_{e_id}_bit_{bit}"
                        for bit in range(env.unit_type_bits)
                    ]
                )

        # Ally features.
        # From the perspective of agent 0.
        al_ids = [al_id for al_id in range(env.n_agents) if al_id != 0]
        for al_id in al_ids:
            feature_names.extend(
                [
                    f"ally_visible_{al_id}",
                    f"ally_distance_{al_id}",
                    f"ally_relative_x_{al_id}",
                    f"ally_relative_y_{al_id}",
                ]
            )
            if env.obs_all_health:
                feature_names.append(f"ally_health_{al_id}")
                if env.shield_bits_ally > 0:
                    feature_names.append(f"ally_shield_{al_id}")
           
            if env.unit_type_bits > 0 and ((not env.replace_teammates or env.observe_teammate_types) or env.zero_pad_unit_types):
                feature_names.extend(
                    [
                        f"ally_unit_type_{al_id}_bit_{bit}"
                        for bit in range(env.unit_type_bits)
                    ]
                )
            if env.obs_last_action:
                feature_names.extend(
                    [
                        f"ally_last_action_{al_id}_action_{action}"
                        for action in range(env.n_actions)
                    ]
                )

        # Own features.
        if env.obs_own_health:
            feature_names.append("own_health")
            if env.shield_bits_ally > 0:
                feature_names.append("own_shield")
        
        if env.unit_type_bits > 0:
            feature_names.extend(
                [
                    f"own_unit_type_bit_{bit}"
                    for bit in range(env.unit_type_bits)
                ]
            )
       

        if env.obs_timestep_number:
            feature_names.append("timestep")

        return feature_names












if __name__ == "__main__":
    
    main(scenarios=['3m', '8m'])
