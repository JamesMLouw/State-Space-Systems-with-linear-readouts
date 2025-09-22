#%%
import os

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import numpy as np

from utils.datasets import Dataset
from utils.model import ESN, ESNModel, ESNModel_DS, RCN, RCNModel,  RCNModel_DS, progress
import utils.measures as meas
import utils.dynamical_systems as ds

dynamical_system_name = 'lorenz'

if dynamical_system_name == 'lorenz':
    from lorenz.config import config

torch.set_default_dtype(config["TRAINING"]["dtype"])

if not os.path.exists(config["PATH"]):
    os.makedirs(config["PATH"])

RC_type = config["MODEL"]["RC_type"]

if RC_type == 'ESN':
    Network = ESN
    Model = ESNModel
    Model_DS = ESNModel_DS
elif RC_type == 'RCN':
    Network = RCN
    Model = RCNModel
    Model_DS = RCNModel_DS
else:
    print('RC not supported')

tag = RC_type + '_ridge_' + str(config["TRAINING"]["ridge"]) + "_model_"

#%% load data

dataset_train = Dataset(
    num_trajectories = config["DATA"]["n_train"],
    len_trajectories = config["DATA"]["l_trajectories"],
    step = config["DATA"]["step"], 
    dynamical_system_name = config["DATA"]["dynamical_system_name"],
    parameters = config["DATA"]["parameters"],
    initial_points_mean = config["DATA"]["y0"], 
    initial_points_sd = config["DATA"]["initial_points_sd"],
    data_type = config["DATA"]["data_type"],
    method = config["DATA"]["method"],
    load_data = True, 
    data_set_name = 'train',
    normalize_data = config["DATA"]["normalize_data"]
)
shift, scale = dataset_train.shift, dataset_train.scale
dataset_train.save_data()

dataset_test = Dataset(
    num_trajectories = config["DATA"]["n_test"],
    len_trajectories = config["DATA"]["l_trajectories_test"],
    step = config["DATA"]["step"], 
    dynamical_system_name = config["DATA"]["dynamical_system_name"],
    parameters = config["DATA"]["parameters"],
    initial_points_mean = config["DATA"]["y0"], 
    initial_points_sd = config["DATA"]["initial_points_sd"],
    data_type = config["DATA"]["data_type"],
    method = config["DATA"]["method"],
    load_data = config["DATA"]["load_data"], 
    data_set_name = 'test',
    normalize_data = config["DATA"]["normalize_data"],
    shift = shift,
    scale = scale
)
dataset_test.save_data()

#%% load model
network = Network(
     config["MODEL"]["input_size"],
    config["MODEL"]["reservoir_size"],
    config["MODEL"]["hidden_size"],
    config["MODEL"]["input_size"],
    config["MODEL"]["scale_rec"],
    config["MODEL"]["scale_in"],
    config["MODEL"]["leaking_rate"],
)

model = Model(
    dataloader_train = None,
    dataloader_val = None,
    network = network,
)

model_name = config["PATH"] + tag
model.load_network(model_name)

#%% predict
warmup = config["DATA"]["max_warmup"]
predictions, _ = model.integrate(
    torch.tensor(dataset_test.input_data[:, :warmup, :], dtype=torch.get_default_dtype()).to(model.device),
    T=dataset_test.input_data.shape[1] - warmup,
)

#%%

batch, T, d = predictions.shape
true_trajs = dataset_test.output_data
pred_trajs = predictions.detach().cpu().numpy()

nu1_indices = []
nu2_indices = []

# separate trajectories based on their first coordinate when warmup ends
for i in range(batch):
    z = true_trajs[i,warmup-1,:]
    if z[0] >=0:
        nu1_indices.append(i)
    else:
        nu2_indices.append(i)

n_nu1 = len(nu1_indices)
n_nu2 = len(nu2_indices)

nu1_trajs_true = np.zeros((n_nu1, T, d))
nu1_trajs_pred = np.zeros((n_nu1, T, d))
nu2_trajs_true = np.zeros((n_nu2, T, d))
nu2_trajs_pred = np.zeros((n_nu2, T, d))

nu1_trajs_true = true_trajs[nu1_indices, :, :]
nu1_trajs_pred = pred_trajs[nu1_indices, :, :]
nu2_trajs_true = true_trajs[nu2_indices, :, :]
nu2_trajs_pred = pred_trajs[nu2_indices, :, :]

# save trajectories

folder = dynamical_system_name + "/predict"
os.makedirs(folder, exist_ok=True)  # creates the folder if it doesn't exist
np.save(os.path.join(folder, "nu1_trajs_true"+ tag), nu1_trajs_true)
np.save(os.path.join(folder, "nu1_trajs_pred"+ tag), nu1_trajs_pred)
np.save(os.path.join(folder, "nu2_trajs_true"+ tag), nu2_trajs_true)
np.save(os.path.join(folder, "nu2_trajs_pred"+ tag), nu2_trajs_pred)


#%%
num_bins = 100
nu1 = nu1_trajs_true[:,warmup-1,:]
nu2 = nu2_trajs_true[:,warmup-1,:]

meas.plot_measure(nu2, (0,2), num_bins, 'hist')
#%% calculate distance between distributions for ESN trajectories
sigma_kernel = 1

dist_trajs_truepred_11 = meas.mmd_rbf_seq(nu1_trajs_true, nu1_trajs_pred)
fig, ax = plt.figure(), plt.axes()
ax.plot(dist_trajs_truepred_11)

np.save(os.path.join(folder, "dist_trajs_truepred_11"+ tag), dist_trajs_truepred_11)
plt.savefig(os.path.join(folder, "dist_trajs_truepred_11"+ tag))
plt.close()

#%%
dist_trajs_truepred_22 = meas.mmd_rbf_seq(nu2_trajs_true, nu2_trajs_pred)
fig, ax = plt.figure(), plt.axes()
ax.plot(dist_trajs_truepred_22)

np.save(os.path.join(folder, "dist_trajs_truepred_22"+ tag), dist_trajs_truepred_22)
plt.savefig(os.path.join(folder, "dist_trajs_truepred_22"+ tag))
plt.close()

#%%
dist_trajs_truepred_12 = meas.mmd_rbf_seq(nu1_trajs_true, nu2_trajs_pred)
fig, ax = plt.figure(), plt.axes()
ax.plot(dist_trajs_truepred_12)

np.save(os.path.join(folder, "dist_trajs_truepred_12"+ tag), dist_trajs_truepred_12)
plt.savefig(os.path.join(folder, "dist_trajs_truepred_12"+ tag))
plt.close()

#%%
dist_trajs_truepred_21 = meas.mmd_rbf_seq(nu2_trajs_true, nu1_trajs_pred)
fig, ax = plt.figure(), plt.axes()
ax.plot(dist_trajs_truepred_21)

np.save(os.path.join(folder, "dist_trajs_truepred_21"+ tag), dist_trajs_truepred_21)
plt.savefig(os.path.join(folder, "dist_trajs_truepred_21"+ tag))
plt.close()

#%%
nu1 = nu1_trajs_pred[:,warmup,:]
meas.plot_measure(nu1, (0,2), num_bins, "hist")

#%%
nu2 = nu2_trajs_true[:,warmup,:]
meas.plot_measure(nu2, (0,2), num_bins, "hist")

#%%
nu1 = nu1_trajs_pred[:,-1,:]
meas.plot_measure(nu1, (0,2), num_bins, "hist")

#%%
nu2 = nu2_trajs_true[:,-1,:]
meas.plot_measure(nu2, (0,2), num_bins, "hist")

#%%
nu1, nu2 = np.expand_dims(nu1_trajs_true[:, warmup, :], axis=1), np.expand_dims(nu2_trajs_pred[:, warmup, :], axis=1)
d1 = meas.mmd_rbf_seq(nu1, nu2)
print(d1)

#%%
nu1, nu2 = np.expand_dims(nu1_trajs_true[:, -1, :], axis=1), np.expand_dims(nu2_trajs_pred[:, -1, :], axis=1)
d2 = meas.mmd_rbf_seq(nu1, nu2)
print(d2)

#%%
d1b, d2b = dist_trajs_truepred_12[warmup], dist_trajs_truepred_12[-1]
print(d1b, d2b)
#%%
x_plot = nu1[:,0]
y_plot = nu1[:,1]
z_plot = nu1[:,2]

fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(1,1,1, projection='3d')
ax.scatter(x_plot, y_plot, z_plot, s = 5)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

#########################################################################################################
#%% compare the invariant measures of lorenz and ESN
# find invariant measure lorenz

n_init_cond_meas = 200
t_start = 300 # need to start from about 300 to make sure you are on the attractor
t_end = 600 # need a good length to cover the whole attractor, 600 seems sufficient
z0 = np.zeros((3))
sd_meas = 20

find_invar_meas = True
if find_invar_meas:
    lor = ds.lorenz()
    mu = meas.invariant_measure(lor,n_init_cond_meas, t_start, t_end, z0, sd_meas )
    np.save('Lorenz invariant measure', mu)
else:
    mu = np.load('Lorenz invariant measure.npy')
#%% plot invariant measure

meas.plot_measure(mu, (2,1), 100, 'hist') # kde takes long and gives strange plots, figure out how to do this better

#%% invariant measure ESN
N = network.reservoir_size
n_init_cond_meas = 300
t_start = 400 # need to start from about 300 to make sure you are on the attractor
t_end = 800 # need a good length to cover the whole attractor, 600 seems sufficient
x0 = np.zeros((N))
sd_meas = 300
model_ds = Model_DS(model)

find_invar_meas = True
if find_invar_meas:
    zeta_ESN = meas.invariant_measure(model_ds, n_init_cond_meas, t_start, t_end, x0, sd_meas )
    zeta_ESN = torch.from_numpy(zeta_ESN)
    mu_ESN = network.readout(zeta_ESN)
    mu_ESN = mu_ESN.detach().cpu().numpy()
    np.save('Lorenz ESN invariant measure state space', zeta_ESN)
    np.save('Lorenz ESN invariant measure readout', mu_ESN)
else:
    zeta_ESN = np.load('Lorenz ESN invariant measure state space.npy')
    mu_ESN = np.load('Lorenz ESN invariant measure readout.npy')

#%% plot invariant measure

meas.plot_measure(mu_ESN, (1,2), 100, 'hist') # kde takes long and gives strange plots, figure out how to do this better


#%%
x_plot = mu_ESN[:,0]
y_plot = mu_ESN[:,1]
z_plot = mu_ESN[:,2]

fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(1,1,1, projection='3d')
ax.scatter(x_plot, y_plot, z_plot, s = 5)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

#%%
ids_off_attractor = []
ids_on_attractor = []
for i in range(mu_ESN.shape[0]):
    if mu_ESN[i,2]<0:
        ids_off_attractor.append(i)
    else:
        ids_on_attractor.append(i)
print(len(ids_off_attractor))
print(mu_ESN.shape[0])

# %%
mu_ESN_2 = mu_ESN[ids_on_attractor, :]
meas.plot_measure(mu_ESN_2, (1,2), 100, 'hist') # kde takes long and gives strange plots, figure out how to do this better


#%%
x_plot = mu_ESN_2[:,0]
y_plot = mu_ESN_2[:,1]
z_plot = mu_ESN_2[:,2]

fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(1,1,1, projection='3d')
ax.scatter(x_plot, y_plot, z_plot, s = 5)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
# %%
