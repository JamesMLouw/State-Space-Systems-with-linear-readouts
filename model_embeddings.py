#%%
import os

import matplotlib.pyplot as plt
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from utils.datasets import Dataset
from utils.model import ESN, ESNModel, RCN, RCNModel, progress

dynamical_system_name = 'lorenz'

if dynamical_system_name == 'lorenz':
    from lorenz.config import config
    
torch.set_default_dtype(config["TRAINING"]["dtype"])

if not os.path.exists(config["PATH"]):
    os.makedirs(config["PATH"])
    
RC_type = config["MODEL"]["RC_type"]
tag = config["FILE_NAME_TAG"]

if RC_type == 'ESN':
    Network = ESN
    Model = ESNModel
elif RC_type == 'RCN':
    Network = RCN
    Model = RCNModel
else:
    print('RC not supported')

#%%
dataset_train = Dataset(
    num_trajectories = config["DATA"]["n_train"],
    len_trajectories = config["DATA"]["l_trajectories_train"],
    step = config["DATA"]["step"], 
    dynamical_system_name = config["DATA"]["dynamical_system_name"],
    parameters = config["DATA"]["parameters"],
    observations_in = config["DATA"]["observations_in"],
    observations_out = config["DATA"]["observations_out"],
    initial_points_mean = config["DATA"]["y0"], 
    initial_points_sd = config["DATA"]["initial_points_sd"],
    data_type = config["DATA"]["data_type"],
    method = config["DATA"]["method"],
    load_data = config["DATA"]["load_data"], 
    data_set_name = 'train',
    normalize_data = config["DATA"]["normalize_data"]
)
shift_in, scale_in = dataset_train.shift_in, dataset_train.scale_in
shift_out, scale_out = dataset_train.shift_out, dataset_train.scale_out
dataset_train.save_data()

#%%
dataset_val = Dataset(
    num_trajectories = config["DATA"]["n_val"],
    len_trajectories = config["DATA"]["l_trajectories_val"],
    step = config["DATA"]["step"], 
    dynamical_system_name = config["DATA"]["dynamical_system_name"],
    parameters = config["DATA"]["parameters"],
    observations_in = config["DATA"]["observations_in"],
    observations_out = config["DATA"]["observations_out"],
    initial_points_mean = config["DATA"]["y0"], 
    initial_points_sd = config["DATA"]["initial_points_sd"],
    data_type = config["DATA"]["data_type"],
    method = config["DATA"]["method"],
    load_data = config["DATA"]["load_data"], 
    data_set_name = 'validate',
    normalize_data = config["DATA"]["normalize_data"],
    shift_in = shift_in,
    scale_in = scale_in,
    shift_out = shift_out,
    scale_out = scale_out
)
dataset_val.save_data()


#%% plot input data
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(dataset_train.tt[:-1], dataset_train.input_data[0][:, 0], label="u")
# ax.plot(dataset_train.tt[:-1], dataset_train.input_data[0][:, 1], label="v")
# ax.plot(dataset_train.tt[:-1], dataset_train.input_data[0][:, 2], label="w")
ax.set_xlabel("t")
plt.legend()

folder = dynamical_system_name + "/fig"
os.makedirs(folder, exist_ok=True)  # creates the folder if it doesn't exist

plt.savefig(os.path.join(folder, "input_data.pdf"), bbox_inches="tight")
plt.show()

plt.close()

#%% plot output data

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(dataset_train.tt[:-1], dataset_train.output_data[0][:, 0], label="u")
ax.set_xlabel("t")
plt.legend()

folder = dynamical_system_name + "/fig"
os.makedirs(folder, exist_ok=True)  # creates the folder if it doesn't exist

plt.savefig(os.path.join(folder, "output_data.pdf"), bbox_inches="tight")
plt.show()

plt.close()

#%%
# Create PyTorch dataloaders for train and validation data
dataloader_train = DataLoader(
    dataset_train,
    batch_size=config["TRAINING"]["batch_size"],
    shuffle=True,
    num_workers=0,
    pin_memory=True,
)
dataloader_val = DataLoader(
    dataset_val,
    batch_size=config["TRAINING"]["batch_size"],
    shuffle=False,
    num_workers=0,
    pin_memory=True,
)

network_sizes = [10, 50, 100, 200, 500]
Networks = []
Models = []

for N in network_sizes:
    network = Network(
        config["MODEL"]["input_size"],
        N, # config["MODEL"]["reservoir_size"],
        config["MODEL"]["hidden_size"],
        config["MODEL"]["output_size"],
        config["MODEL"]["scale_rec"],
        config["MODEL"]["scale_in"],
        config["MODEL"]["leaking_rate"],
    )
    model = Model(
        dataloader_train,
        dataloader_val,
        network,
        learning_rate=config["TRAINING"]["learning_rate"],
        offset=config["TRAINING"]["offset"],
        ridge_factor=config["TRAINING"]["ridge_factor"],
        device=config["TRAINING"]["device"],
    )
    Networks.append(network)
    Models.append(model)

#%%

for model in Models:
    states_embedding = model.generate_embedding_data()
    print(states_embedding.shape)
    model.save_network_embedding(config["PATH"] + tag + "_model_")

#%%

