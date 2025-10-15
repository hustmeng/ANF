import requests
import torch
from torch.cuda.amp import autocast

from layers import flow as nf
from training import loss
import utils


def make_optimizer(opt_name, opt_par, model):
    return getattr(torch.optim, opt_name)(model["layers"].parameters(), **opt_par)


def make_scheduler(scheduler_name, scheduler_par, optimizer):
    if scheduler_name:
        return getattr(torch.optim.lr_scheduler, scheduler_name)(optimizer, **scheduler_par)


def make_loss_fn(estimator):
    return getattr(loss, estimator)


def step(batch_size, *, model, action, loss_fn, n_batches=1, use_amp) -> (torch.Tensor, torch.Tensor, torch.Tensor):
    logq_list = []
    logp_list = []
    loss_ = 0.0
    prior = model['prior']

    for i in range(n_batches):
        with autocast(enabled=use_amp):
            z = prior.sample_n(batch_size=batch_size)
            log_prob_z = prior.log_prob(z)
        try:
            l, logq, logp = loss_fn(z, log_prob_z, model=model, action=action, use_amp=use_amp)
        except AssertionError:
            raise AssertionError

        l /= n_batches
        l.backward(retain_graph=True)

        loss_ += l.detach()
        logq_list.append(logq)
        logp_list.append(logp)

    with torch.no_grad():
        logq = torch.cat(logq_list)
        logp = torch.cat(logp_list)

    return loss_, logq, logp

def calc_dkl(logp, logq):
    return (logq - logp).mean() # reverse KL, assuming samples from q

def train_step(*, model, action, loss_fn, batch_size, optimizer, scheduler=None, n_batches=1, use_amp):
    optimizer.zero_grad(set_to_none=True)

    loss_, logq, logp = step(
        batch_size=batch_size,
        model=model,
        action=action,
        loss_fn=loss_fn,
        n_batches=n_batches,
        use_amp=use_amp
    )
    
    optimizer.step()

    if scheduler is not None:
        scheduler.step()

    with torch.no_grad():
        dkl = nf.calc_dkl(logp, logq)
        std_dkl = (logq - logp).std()
        ess = nf.compute_ess(logp, logq)
    return {
        "ess": utils.grab(ess).item(),
        "loss": loss_.detach().cpu().numpy(),
        "dkl": utils.grab(dkl).item(),
        "std_dkl": utils.grab(std_dkl).item()
    }
