import torch
from torch.cuda.amp import autocast

from layers import flow as nf


def REINFORCE_loss(z_a, log_prob_z_a, *, model, action, use_amp):
    layers, prior = model['layers'], model['prior']
    with torch.no_grad():
        with autocast(enabled=use_amp):
            phi, logq = nf.apply_flow(layers, z_a, log_prob_z_a)
            logp = -action(phi)
            signal = logq - logp

    with autocast(enabled=use_amp):
        z, log_q_phi = nf.reverse_apply_flow(layers, phi, torch.zeros_like(log_prob_z_a, device=phi.device))
        prob_z = prior.log_prob(z)
        log_q_phi = prob_z - log_q_phi
        loss = torch.mean(log_q_phi * (signal - signal.mean()))

    return loss, logq, logp


def rt_loss(z, log_prob_z, *, model, action, use_amp):
    layers = model['layers']

    with autocast(enabled=use_amp):
        x, logq = nf.apply_flow(layers, z, log_prob_z)

        logp = -action(x)
        loss = nf.calc_dkl(logp, logq)

        return loss, logq.detach(), logp.detach()
