def team_composition_label_to_list(team_comp_label):
    
    agents_per_class = team_comp_label.split('_') # split per class
    
    team_comp = []
    
    for class_agents in agents_per_class:
        
        num_agents = int(class_agents[0])
        agents_class = class_agents[1:]
        
        if agents_class == 'ma':
            for _ in range(num_agents):
                team_comp.append('marine')
            
        elif agents_class == 'mr':
            for _ in range(num_agents):
                team_comp.append('marauder')
        
        elif agents_class == 'md':
            for _ in range(num_agents):
                team_comp.append('medivac')
    
    
        # support for zerg units
        elif agents_class == 'z':
            for _ in range(num_agents):
                team_comp.append('zergling')
                
        elif agents_class == 'b':
            for _ in range(num_agents):
                team_comp.append('baneling')
        
        elif agents_class == 'h':
            for _ in range(num_agents):
                team_comp.append('hydralisk')
                
                
        # support for protoss units
        elif agents_class == 'st':
            for _ in range(num_agents):
                team_comp.append('stalker')
                
        elif agents_class == 'ze':
            for _ in range(num_agents):
                team_comp.append('zealot')
        
        elif agents_class == 'co':
            for _ in range(num_agents):
                team_comp.append('colossus')     
                
                
            
    return team_comp


def return_distribution_config():
    

    distribution_configs = {
        "terran_5_vs_5": {
            "n_units": 5,
            "n_enemies": 5,
            "team_gen": {
                "dist_type": "weighted_teams",
                "unit_types": ["marine", "marauder", "medivac"],
                "weights": [0.45, 0.45, 0.1],
                "observe": True,
            },
            
            
            
            "start_positions": {
                "dist_type": "surrounded_and_reflect",
                "p": 0.5,
                "n_enemies": 5,
                "map_x": 32,
                "map_y": 32,
            },
        },
        
        "zerg_5_vs_5": {
            "n_units": 5,
            "n_enemies": 5,
            "team_gen": {
                "dist_type": "weighted_teams",
                "unit_types": ["zergling", "baneling", "hydralisk"],
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
        },
        
        "protoss_5_vs_5": {
            "n_units": 5,
            "n_enemies": 5,
            "team_gen": {
                "dist_type": "weighted_teams",
                "unit_types": ["stalker", "zealot", "colossus"],
                "weights": [0.45, 0.45, 0.1],
                "observe": True,
            },
            "start_positions": {
                "dist_type": "surrounded_and_reflect",
                "p": 0.5,
                "n_enemies": 5,
                "map_x": 32,
                "map_y": 32,
            },
        },
        
        "terran_10_vs_10": {
            "n_units": 10,
            "n_enemies": 10,
            "team_gen": {
                "dist_type": "weighted_teams",
                "unit_types": ["marine", "marauder", "medivac"],
                "exception_unit_types": ["baneling"],
                "weights": [0.45, 0.45, 0.1],
                "observe": True,
            },
            
        
            "start_positions": {
                "dist_type": "surrounded_and_reflect",
                "p": 0.5,
                "n_enemies": 5,
                "map_x": 32,
                "map_y": 32,
            },
        },
    }
    
    
    ####################################################################################################################
    ### terran subteams
        
    terrran_5_vs_5_compositions = [
        '1ma_1mr_3md', 
        '1ma_2mr_2md', 
        '1ma_3mr_1md', 
        '1ma_4md', 
        '1ma_4mr', 
        '1mr_4md', 
        '2ma_1mr_2md', 
        '2ma_2mr_1md', 
        '2ma_3md', 
        '2ma_3mr', 
        '2mr_3md', 
        '3ma_1mr_1md', 
        '3ma_2md', 
        '3ma_2mr', 
        '3mr_2md', 
        '4ma_1md', 
        '4ma_1mr', 
        '4mr_1md', 
        '5ma', 
        '5mr'
    ]
    
    for team_comp in terrran_5_vs_5_compositions: 
         
        distribution_configs[f'terran_5_vs_5_{team_comp}'] = {
            
            "n_units": 5,
            "n_enemies": 5,
            
            "team_gen": {
                "dist_type": "fixed",
                
                # "items": [["marine", "marauder", "medivac", "medivac", "medivac"]],
                "items": [team_composition_label_to_list(team_comp)],
                
                "test_mode": True,
                "observe": True,
                },
            
            "start_positions": {
                "dist_type": "surrounded_and_reflect",
                "p": 0.5,
                "map_x": 32,
                "map_y": 32,
                },
            }
        
    ####################################################################################################################
    ### zerg subteams
    
    zerg_5_vs_5_compositions = [
        '1z_1h_3b', 
        '1z_2h_2b', 
        '1z_3h_1b', 
        '1z_4b', 
        '1z_4h', 
        '1h_4b', 
        '2z_1h_2b', 
        '2z_2h_1b', 
        '2z_3b', 
        '2z_3h', 
        '2h_3b', 
        '3z_1h_1b', 
        '3z_2b', 
        '3z_2h', 
        '3h_2b', 
        '4z_1b', 
        '4z_1h', 
        '4h_1b', 
        '5z', 
        '5h'
    ]

    
    for team_comp in zerg_5_vs_5_compositions: 
        
        distribution_configs[f'zerg_5_vs_5_{team_comp}'] = {
            
            "n_units": 5,
            "n_enemies": 5,
            
            "team_gen": {
                "dist_type": "fixed",
                
                "items": [team_composition_label_to_list(team_comp)],
                
                "test_mode": True,
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
        
    protoss_5_vs_5_compositions = [
        '1st_1ze_3co', 
        '1st_2ze_2co', 
        '1st_3ze_1co', 
        '1st_4co', 
        '1st_4ze', 
        '1ze_4co', 
        '2st_1ze_2co', 
        '2st_2ze_1co', 
        '2st_3co', 
        '2st_3ze', 
        '2ze_3co', 
        '3st_1ze_1co', 
        '3st_2co', 
        '3st_2ze', 
        '3ze_2co', 
        '4st_1co', 
        '4st_1ze', 
        '4ze_1co', 
        '5st', 
        '5ze'
    ]

    
    for team_comp in protoss_5_vs_5_compositions: 
        
        distribution_configs[f'protoss_5_vs_5_{team_comp}'] = {
            
            "n_units": 5,
            "n_enemies": 5,
            
            "team_gen": {
                "dist_type": "fixed",
                
                "items": [team_composition_label_to_list(team_comp)],
                
                "test_mode": True,
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
        
    return distribution_configs