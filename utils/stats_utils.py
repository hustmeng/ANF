import numpy as np
import torch


def _numpy_block(x, binsize):
    n_bins = len(x) // binsize
    xb = np.stack(np.array_split(x, n_bins), axis=0)
    return xb


def block_mean(sample, block_size):
    xb = _numpy_block(sample, block_size)
    return np.mean(xb, axis=1)


def ac(series, history=100):
    """Autocorrelation function for given series

    Parameters
    ----------
    series : array_like
        Array containing series whose autocorrelation function is to be computed
    history
        how much history to use in the autocorrelation function
    Returns
    -------
        Autocorrelation function : ndarray
    """
    mu = np.mean(series)
    var = np.mean((series - mu) * (series - mu))
    out = np.empty(history)
    for i in range(history):
        out[i] = np.mean((series[:-history] - mu) * (series[i: i - history] - mu))
    return out / var


def ac_and_tau_int(series, c=10, maxlen=200):
    mu = np.mean(series)
    var = np.mean((series - mu) * (series - mu))
    out = [1.0]
    tau_int = 0.5
    for t in range(1, maxlen):
        cor = np.mean((series[:-t] - mu) * (series[t:] - mu)) / var
        tau_int += cor
        out.append(cor)
        if t > c * tau_int:
            break
    return tau_int, np.asarray(out)


def list_mean(data, mean=0.0):
    mean = mean
    size = 0
    for t in data:
        mean += t.sum()
        size += t.nelement()

    return mean / size


def bootstrapf(f, x, *, n_samples, binsize):
    n_bins = len(x) // binsize
    xb = np.stack(np.array_split(x, n_bins), axis=0)

    boots = []

    for i in range(n_samples):
        bsample = np.resize(xb[np.random.choice(n_bins, n_bins)], (n_bins * binsize, *x.shape[1:]))
        boots.append(f(bsample))
    return np.mean(boots, 0), np.std(boots, 0)


def bootstrap(x, *, n_samples, binsize):
    return bootstrapf(lambda x: np.mean(x, 0), x, n_samples=n_samples, binsize=binsize)


def _torch_block(x, binsize):
    xblocks = torch.split(x, binsize, dim=0)

    if len(xblocks[0]) != len(xblocks[-1]):
        xblocks = xblocks[:-1]

    n_bins = len(xblocks)

    xb = torch.stack(xblocks, dim=0)

    return xb


def torch_bootstrapf(f, x, *, n_samples, binsize):
    xb = _torch_block(x, binsize)
    n_bins = len(xb)

    boots = []

    for i in range(n_samples):
        idx = torch.from_numpy(np.random.choice(n_bins, n_bins))
        bsample = xb[idx].view(
            (n_bins * binsize, *x.shape[1:])
        )
        boots.append(f(bsample))
    boots = torch.stack(boots, dim=0)
    return torch.mean(boots, 0), torch.std(boots, 0)


def torch_bootstrap_mean(x, *, n_samples, binsize, logweights=None):
    xb = _torch_block(x, binsize)
    n_bins = len(xb)

    boots = []
    if logweights is not None:
        weights = torch.exp(logweights - torch.max(logweights))
        bweights = _torch_block(weights, binsize)

    for i in range(n_samples):
        idx = torch.from_numpy(np.random.choice(n_bins, n_bins))
        bsample = xb[idx].view(
            (n_bins * binsize, *x.shape[1:])
        )
        if logweights is not None:
            bwsample = bweights[idx].ravel()
            norm = torch.mean(bwsample)
            boots.append(torch.mean(bwsample * bsample / norm, 0))
        else:
            boots.append(torch.mean(bsample, 0))
    boots = torch.stack(boots, dim=0)
    return torch.mean(boots, 0), torch.std(boots, 0)


def torch_bootstrapo(obs, x, *, n_samples, binsize, logweights=None):
    obs_ = obs(x)
    return torch_bootstrap_mean(x=obs_, n_samples=n_samples, binsize=binsize, logweights=logweights)
