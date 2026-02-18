import pickle

file_path = "/workspace/CFCQL/discrete/offline_datasets/demos_10x10_ocean_3keep_3hstep/trajectory_0.pkl" 

with open(file_path, "rb") as f:
    data = pickle.load(f)
    
b = 1