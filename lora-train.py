#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Fine turning phi^4 model

"""

import argparse
import time
import sys
import json

from numpy import log
import torch
import torch.nn as nn
import training.loss
from training.train import train_step

import utils
import utils.metrics as metrics
import utils.checkpoint as chck

import layers.phi4 as phi4
from layers.affine_couplings import make_phi4_affine_layers
import layers.flow as nf
import utils.scripts as scripts
from utils.stats_utils import torch_bootstrapf, torch_bootstrapo
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
parser.add_argument('-L', '--lattice-size', type=int, action='store', default= 12,
                    help='L - lattice size will be LxL')
parser.add_argument('--batch-size', type=int, action='store', default='256', help='Size of the batch')
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
parser.add_argument('-l', '--lambda', type=float, dest='lamda', default='5.5',
                    help='lambda parameter of the action')
parser.add_argument('-lr', '--learning-rate', type=float, default='0.005',        
                    help='Learning rate for the Adam optimizer')
parser.add_argument('--n-eras', type=int, default=5000, help='Number of eras')
parser.add_argument('--n-epochs-per-era', type=int, default=200,
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

model_cfg = {'n_layers': 32,
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

checkpoint = torch.load("phi4_rt_12x12_base.zip", map_location=torch_device)
state_dict = checkpoint["state_dict"]

layers.load_state_dict(state_dict)

import torch.nn as nn

total_params = 0
for name, module in layers.named_modules():
    if isinstance(module, (nn.Conv2d, nn.Linear,nn.ConvTranspose2d,)):
        for param_name, param in module.named_parameters():
                   
             if torch.is_floating_point(param):
                        print(f"Adding noise to: {name}")

                        conductance_min = 20 
                        conductance_max = 80
                        conductance_zero = 50 
                        std = 0.8

                        max_val = param.data.max().item()
                        min_val = param.data.min().item()
                        pos_max = max(0, max_val)
                        neg_max = abs(min(0, min_val))
                        if pos_max > neg_max:
                           scale = (conductance_max - conductance_zero) / pos_max
                        else:
                           scale = (conductance_max - conductance_zero) / neg_max

                        conductance = param.data * scale + conductance_zero
                        noise = torch.randn_like(conductance) * std
                        conductance = conductance + noise
                        param.data = (conductance - conductance_zero) / scale

                        param.requires_grad = False

                        total_params += param.numel()
                        print(param.data.min(), param.data.max())


from layers.lora import get_updated_model
layers = get_updated_model(layers, rank = 2, alpha = 1.0,  device=torch_device) # update the model

model = {'layers': layers, 'prior': prior}
print(model)

trainable_params = sum(p.numel() for p in layers.parameters() if p.requires_grad)
print(f"需要训练的权重数量: {trainable_params}")


logger.info(model)
print_freq = 20 # epochs

history = {
    'dkl': [],
    'std_dkl': [],
    'loss': [],
    'ess': []
}

optimizer = torch.optim.Adam(model['layers'].parameters(), lr=args.learning_rate)

elapsed_time = 0
start_time = time.time()

total_epochs = args.n_eras * args.n_epochs_per_era
epochs_done = 0
if args.verbose > 0:
    print(f"Starting training: {args.n_eras} x {args.n_epochs_per_era} epochs")
    logger.info(f"Starting training: {args.n_eras} x {args.n_epochs_per_era} epochs")
for era in range(args.n_eras):
    for epoch in range(args.n_epochs_per_era):
        m = train_step(use_amp=False, model=model, action=action, loss_fn=loss_function, batch_size=args.batch_size,
                       n_batches=args.n_batches, optimizer=optimizer)
        metrics.add_metrics(history, m)
        epochs_done += 1
        if (epoch + 1) % print_freq == 0:
            chck.safe_save_checkpoint(model=layers, optimizer=optimizer, scheduler=None, era=era, model_cfg=model_cfg,
                                      **{'mass2': args.mass2, 'lambda': args.lamda},
                                      path=f"phi4_{args.loss}_{L:02d}x{L:02d}_lora.zip")
            elapsed_time = time.time() - start_time
            avg = metrics.average_metrics(history, args.n_epochs_per_era, history.keys())

            print(f"Finished era {era + 1:d} epoch {epoch + 1:d} elapsed time {elapsed_time:.1f}", end="")
            logger.info(f"Finished era {era + 1:d} epoch {epoch + 1:d} elapsed time {elapsed_time:.1f}")
            if epochs_done > 0:
                time_per_epoch = elapsed_time / epochs_done
                time_remaining = (total_epochs - epochs_done) * time_per_epoch
                if args.verbose > 0:
                    print(f"  {time_per_epoch:.2f}s/epoch  remaining time {utils.format_time(time_remaining):s}")
                    logger.info(f"  {time_per_epoch:.2f}s/epoch  remaining time {utils.format_time(time_remaining):s}")
                    metrics.print_dict(avg,logger=logger)
                    #for key, val in avg.items():
                    #logger.info(avg)

print(model)

if args.verbose > 0:
    print(f"{elapsed_time / args.n_eras:.2f}s/era")

if args.n_samples > 0:
    print(f"Sampling {args.n_samples} configurations")
    if args.mass2 > 0.0:
        F_exact = phi4.free_field_free_energy(L, args.mass2)
    u, lq = nf.sample(batch_size=batch_size, n_samples=args.n_samples, prior=prior, layers=layers)
    lp = -action(u)
    lw = lp - lq
    F_q, F_q_std = torch_bootstrapf(lambda x: -torch.mean(x), lw, n_samples=args.n_boot_samples, binsize=args.bin_size)

    lw = lp - lq
    F_nis, F_nis_std = torch_bootstrapf(lambda x: -(torch.special.logsumexp(x, 0) - log(len(x))), lw,
                                        n_samples=args.n_boot_samples,
                                        binsize=args.bin_size)
    if args.lamda == 0.0:
        print(f"Variational free energy = {F_q:.3f}+/-{F_q_std:.4f} diff = {F_q - F_exact:.4f}")
        print(f"NIS free energy = {F_nis:.3f}+/-{F_nis_std:.4f} diff = {F_nis - F_exact:.4f}")
    else:
        print(f"Variational free energy = {F_q:.3f}+/-{F_q_std:.4f}")
        print(f"NIS free energy = {F_nis:.3f}+/-{F_nis_std:.4f}")

    mag2, mag2_std = torch_bootstrapo(lambda x: torch.sum(x, dim=(1, 2)) ** 2 / (L * L), u, n_samples=100, binsize=16,
                                      logweights=lw)
    print(f"Magnetization^2 /(L*L) = {mag2.mean():.3f}+/-{mag2_std.mean():.4f}")

