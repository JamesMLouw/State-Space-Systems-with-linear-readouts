#%%
import os

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import numpy as np

from utils.datasets import Dataset, Two_Sample
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
tag = config["FILE_NAME_TAG"]

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

model_name = config["PATH"] + tag + "_model_"
model.load_network(model_name)

#%%
load_samples = config["DATA"]["load_samples"]
load_sample_dists = config["DATA"]["load_sample_dists"]

#%% predict
warmup = config["DATA"]["max_warmup"]

if not load_samples:
    predictions, _ = model.integrate(
        torch.tensor(dataset_test.input_data[:, :warmup, :], dtype=torch.get_default_dtype()).to(model.device),
        T=dataset_test.input_data.shape[1] - warmup,
    )

#%%
folder = dynamical_system_name + "/predict"
    
if load_samples:
    nu1_trajs_true = None
    nu1_trajs_pred = None
    nu2_trajs_true = None
    nu2_trajs_pred = None
else:
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

    nu1_trajs_true = true_trajs[nu1_indices, :, :]
    nu1_trajs_pred = pred_trajs[nu1_indices, :, :]
    nu2_trajs_true = true_trajs[nu2_indices, :, :]
    nu2_trajs_pred = pred_trajs[nu2_indices, :, :]

    # save trajectories
    os.makedirs(folder, exist_ok=True)  # creates the folder if it doesn't exist
    np.save(os.path.join(folder, "nu1_trajs_true"+ tag), nu1_trajs_true)
    np.save(os.path.join(folder, "nu1_trajs_pred"+ tag), nu1_trajs_pred)
    np.save(os.path.join(folder, "nu2_trajs_true"+ tag), nu2_trajs_true)
    np.save(os.path.join(folder, "nu2_trajs_pred"+ tag), nu2_trajs_pred)


#%%
if not load_samples:
    num_bins = 100
    nu1 = nu1_trajs_true[:,warmup-1,:]
    nu2 = nu2_trajs_true[:,warmup-1,:]

    meas.plot_measure(nu2, (0,2), num_bins, 'hist')
#%%
print(folder)
#%% calculate distance between distributions for trajectories
name = "dist_trajs_truepred_12" + tag + "_model_"
name1 = "nu1_trajs_true" + tag + "_model_"
name2 = "nu2_trajs_pred" + tag + "_model_"
samples_12 = Two_Sample(nu1_trajs_true,
                        nu2_trajs_pred, 
                        load_samples,
                        load_sample_dists,
                        folder + "/", 
                        name, 
                        name1,
                        name2)

if not load_sample_dists:
    sigma_kernel = samples_12.median_dist(100)
    print(f"median distance between points (averaged over time) is {sigma_kernel}")
    samples_12.calculate_dist(sigma = sigma_kernel, biased = True,
                               linear_time = False, enforce_equal=False)
    samples_12.calculate_dist(sigma = sigma_kernel, biased = False,
                               linear_time = False, enforce_equal=True)
    samples_12.calculate_dist(sigma = sigma_kernel, biased = False,
                               linear_time = True, enforce_equal=True)


#%% calculate distance between distributions for trajectories
name = "dist_trajs_truepred_11" + tag + "_model_"
name1 = "nu1_trajs_true" + tag + "_model_"
name2 = "nu1_trajs_pred" + tag+ "_model_"
samples_11 = Two_Sample(nu1_trajs_true,
                        nu1_trajs_pred, 
                        load_samples,
                        load_sample_dists,
                        folder + "/", 
                        name, 
                        name1,
                        name2)

if not load_sample_dists:
    samples_11.calculate_dist(sigma = sigma_kernel, biased = True,
                               linear_time = False, enforce_equal=False)
    samples_11.calculate_dist(sigma = sigma_kernel, biased = False,
                               linear_time = False, enforce_equal=True)
    samples_11.calculate_dist(sigma = sigma_kernel, biased = False,
                               linear_time = True, enforce_equal=True)

###############################################################################
###############################################################################
###############################################################################
# dist_trajs_truepred_11 = samples_11.dist
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

def two_sample_test(m, alpha = 0.05, H0 = '===', epsilon = None, biased = True):
    """
    Test to differentiate between samples from two distributions.

    Params
        m : sample size (equal samples)
        alpha : p-value
        H0 : null hypothesis, options: '==', '>eps', '<eps'
        epsilon : test value in case null is not '=='
        biased : biased or unbiased statistic

    Returns
        squared critical value for acceptance/rejection
    """

def sample_size_two_sample_test(l_sq, h_sq, alpha = 0.05, biased = True):
    """
    Sample size required to perform a two sample test
    to differentiate between samples from two distributions.

    Params
        l_sq, h_sq : low and high values (squared) to differentiate
        alpha : p-value
        biased : biased or unbiased statistic

    Returns
        sample size required to differentiate between high and low values
    """

