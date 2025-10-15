#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Inference phi^4 model

"""

import argparse
import time
import sys
import json
import numpy as np
from numpy import log
import torch

import training.loss
from training.train import train_step

import utils
import utils.metrics as metrics
import utils.checkpoint as chck

import layers.phi4 as phi4
from layers.phi4 import correlation_length, two_point_central
from layers.affine_couplings import make_phi4_affine_layers
import layers.flow as nf
import utils.scripts as scripts
from utils.stats_utils import torch_bootstrapf, torch_bootstrapo
from layers.mcmc import make_mcmc_ensemble

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

seed = 42
torch.manual_seed(seed)

import logging
import time

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='training_log.log',
                    filemode='w')
logger = logging.getLogger()


parser = argparse.ArgumentParser(description=__doc__, formatter_class=scripts.RawTextArgumentDefaultsHelpFormatter)

parser.add_argument('--device', type=str, default='cuda', help='Device to run the training on.')
parser.add_argument('--list-cuda-devices', action='store_true', help="Lists cuda devices and exists")
parser.add_argument('-L', '--lattice-size', type=int, action='store', default=12,
                    help='L - lattice size will be LxL')
parser.add_argument('--batch-size', type=int, action='store', default='1024', help='Size of the batch')
parser.add_argument('--n-batches', '-n', type=int, default='1',
                    help='Number of batches used for one gradient update')
parser.add_argument('--configuration', action='store', help='Configuration file')
parser.add_argument('-v', '--verbose', type=int, default='1',
                    help='Verbosity level if zero does not print anything.')
parser.add_argument('--loss', default='rt', choices=['REINFORCE', 'rt'], help='Loss function')
parser.add_argument('--float-dtype', default='float32', choices=['float32', 'float64'],
                    help='Float precision used for training')
parser.add_argument('-m2', '--mass2', type=float, default='-4.0',
                    help='m^2 mass squared parameter of the action')
parser.add_argument('-l', '--lambda', type=float, dest='lamda', default='5.0',
                    help='lambda parameter of the action')
parser.add_argument('-lr', '--learning-rate', type=float, default='0.001',
                    help='Learning rate for the Adam optimizer')
parser.add_argument('--n-eras', type=int, default=10000, help='Number of eras')
parser.add_argument('--n-epochs-per-era', type=int, default=100,
                    help='Numbers of gradient updates per era')

parser.add_argument('--n-samples', type=int, default=2 ** 16,
                    help='Number of samples used for evaluation')
parser.add_argument('--n-boot-samples', type=int, default=100,
                    help='Number of bootstrap samples')
parser.add_argument('--bin-size', type=int, default=16,
                    help='Bin size for bootstrap')

args = parser.parse_args()

if args.list_cuda_devices:
    scripts.list_cuda_devices()
    sys.exit(0)

if args.verbose > 0:
    print(f"Running on PyTorch {torch.__version__}")

torch_device = args.device
scripts.check_cuda(torch_device)

if args.verbose:
    scripts.describe_device(torch_device)

batch_size = args.batch_size
float_dtype = args.float_dtype

L = args.lattice_size
lattice_shape = (L, L)

action = phi4.ScalarPhi4Action(args.mass2, args.lamda)
loss_function = getattr(training.loss, f"{args.loss}_loss")

model_cfg = {'n_layers': 20,
             'hidden_feature': 18,
             'patch_size':3,
             'depth': 5,
             'lattice_shape': lattice_shape
             }

if args.configuration:
    with open(args.configuration) as f:
        model_cfg = json.load(f)

layers = make_phi4_affine_layers(**model_cfg, device=torch_device)
prior = nf.SimpleNormal(torch.zeros(lattice_shape).to(device=torch_device),
                        torch.ones(lattice_shape).to(device=torch_device))


checkpoint = torch.load("phi4_rt_12x12_lora.zip", map_location=torch_device, weights_only=True)
state_dict = checkpoint["state_dict"]



model = {'layers': layers.to(torch_device), 'prior': prior}

from layers.lora import get_updated_model
layers = get_updated_model(layers, rank = 2, alpha = 1.0,  device=torch_device) # update the model

print(layers)
layers.load_state_dict(state_dict)

model = {'layers': layers.to(torch_device), 'prior': prior}

ensemble_size = 128 * 2048
phi4_ens = make_mcmc_ensemble(model, action, 2048, ensemble_size)
print("Accept rate:", np.mean(phi4_ens['accepted']))

def grab(var):
    if isinstance(var, torch.Tensor):
        return var.detach().cpu().numpy()
    elif isinstance(var, np.ndarray):
        return var


n_therm = ensemble_size // 2

cfgs = np.stack(list(map(grab, phi4_ens['x'])), axis=0)[n_therm:]
C = 0
for x in range(L):
    for y in range(L):
        C = C + cfgs*np.roll(cfgs, (-x, -y), axis=(1,2))
X = np.mean(C, axis=(1,2))

#np.save("cfgs.npy", cfgs)

def bootstrap(x, *, Nboot, binsize):
    boots = []
    x = x.reshape(-1, binsize, *x.shape[1:])
    for i in range(Nboot):
        boots.append(np.mean(x[np.random.randint(len(x), size=len(x))], axis=(0,1)))
    return np.mean(boots), np.std(boots)

X_mean, X_err = bootstrap(X, Nboot=100, binsize=4)
print(f'Two-point susceptibility = {X_mean:.2f} +/- {X_err:.2f}')

def compute_correlation_function(cfgs):
    lattice_size = cfgs.shape[1]
    C = np.zeros((lattice_size, lattice_size))

    for x in range(lattice_size):
        for y in range(lattice_size):
            product = cfgs * np.roll(cfgs, (x, y), axis=(1, 2))
            mean_product = np.mean(product)

            mean_phi_x = np.mean(cfgs)
            mean_phi_x_y = np.mean(np.roll(cfgs, (x, y), axis=(1, 2)))

            C[x, y] = mean_product - mean_phi_x * mean_phi_x_y
    return C

def compute_average_ising_energy_density(G_c, d=2):
    E = 0
    single_site_displacements = [(1, 0), (0, 1)]

    for displacement in single_site_displacements:
        x, y = displacement
        E += G_c[x, y]
    return E / d

def mp(Gc_0):
    arg = (np.roll(Gc_0, 1) + np.roll(Gc_0, -1)) / (2 * Gc_0)
    mp = np.arccosh(arg[1:])
    return mp

def correlation_length(G):
    """Estimator for the correlation length.

    Args:
        G: Centered two-point function.

    Returns:
        Scalar. Estimate of correlation length.
    """
    #print(G.shape)
    Gs = np.mean(G, axis=0)
    arg = (np.roll(Gs, 1) + np.roll(Gs, -1)) / (2 * Gs)
    mp = np.arccosh(arg[1:])
    return 1 / np.nanmean(mp)

two_point = compute_correlation_function(cfgs)
Gc_0 = np.mean(two_point,axis = 0)
print("Green function:",Gc_0)

m_p = mp(Gc_0)
print("mp:",m_p)

def compute_two_point_susceptibility(G_c):
    chi_2 = np.sum(G_c)
    return chi_2

chi_2 = compute_two_point_susceptibility(two_point)  # 计算二阶易敏度

print("susceptibility:", chi_2)


corr_len = correlation_length(two_point)

print("correlation_length:",lattice_shape[0] / corr_len)

E = compute_average_ising_energy_density(two_point)  # 计算平均Ising能量密度
print("Ising Energy:",E)
