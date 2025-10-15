# This is modified code from https://arxiv.org/abs/2101.08176 by M.S. Albergo et all.
"""Various normalizing flow utilities."""

import torch
from utils.calc_function import calc_function
import torch.nn as nn

class SimpleNormal:
    """Normal prior distribution

    At each site the distribution is an independent  normal distribution.
    """

    def __init__(self, loc, std):
        """

        Parameters
        ----------
        loc: torch.Tensor
            mean of the distribution
        std: torch.Tensor
            standard deviation of the distribution
        """
        self.dist = torch.distributions.normal.Normal(torch.flatten(loc), torch.flatten(std))
        self.shape = loc.shape

    def log_prob(self, x):
        """Log probability of the distribution  for given configurations

        Parameters
        ----------
        x : torch.Tensor
            configurations
        Returns
        -------
            log probability of the prior distribution for each configuration
        """

        logp = self.dist.log_prob(x.reshape(x.shape[0], -1))
        return torch.sum(logp, dim=1)

    def sample_n(self, batch_size):
        """Generate samples from the distribution

        Parameters
        ----------
        batch_size
            number of samples to generate
        Returns
        -------
            generated samples
        """

        x = self.dist.sample((batch_size,))
        return x.reshape(batch_size, *self.shape)


def apply_flow(coupling_layers, z_in, logq_a):
    """Apply a normalizing flow to configurations.

    Parameters
    ----------
    coupling_layers
        normalizing flow layers
    z_in
        prior configurations
    logq_a
        log probability of the prior configurations

    Returns
    -------
        Transformed configurations and the log probability of the transformed configurations
    """

    logq = torch.zeros_like(logq_a)
    t = torch.tensor(0).type(torch.float).to(logq.device)
    t = torch.tensor(0).to(logq.device) 
    for layer in coupling_layers:
        try:
            z, logJ = layer.forward(z_in, t)
        except AssertionError:
            # Your debugging code goes here
            raise AssertionError
        logq -= logJ
        z_in = z
        t = t + 1
    return z, logq + logq_a


def apply_flow_to_prior(prior, coupling_layers, *, batch_size):
    """Generates prior configurations and applies a normalizing flow to them.

    Parameters
    ----------
    prior
        distribution to sample prior configurations from
    coupling_layers
        normalizing flow layers
    batch_size
        number of prior configurations to generate

    Returns
        generated configurations and the log probability of the generated configurations
    -------
    """

    z = prior.sample_n(batch_size)
    logq = prior.log_prob(z)
    return apply_flow(coupling_layers, z, logq)


def reverse_apply_flow(coupling_layers, phi, logq_a):
    """Apply the inverse of a normalizing flow to configurations.

    Parameters
    ----------
    coupling_layers
        normalizing flow layers
    phi
        configurations to transform
    logq_a
        log probability of the configurations

    Returns
        Transformed configurations and the log probability of the transformed configurations
    -------

    """
    logq = torch.zeros_like(logq_a)
    t = torch.tensor(0).type(torch.float).to(logq.device)
    for layer in reversed(coupling_layers):
        try:
            phi, logJ = layer.reverse(phi, t)
        except AssertionError:
            # Your debugging code goes here
            raise AssertionError
        logq -= logJ
        t = t + 1
    return phi, logq + logq_a

class ResidualConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, padding_mode, dtype):
        super(ResidualConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              padding=dilation * (kernel_size - 1) // 2,
                              stride=1, padding_mode=padding_mode, dilation=dilation, dtype=dtype, bias = False)
        nn.init.normal_(self.conv.weight, mean=0, std=0.01)
        self.activation = nn.LeakyReLU()

    def forward(self, x):
        out = self.conv(x)
        if self.conv.in_channels == self.conv.out_channels:
            out = out + x
        return self.activation(out)

def make_conv_resnet(*, in_channels, hidden_sizes, out_channels, kernel_size, use_final_tanh, dilation=1,
                  float_dtype=torch.float32):
    sizes = [in_channels] + hidden_sizes + [out_channels]
    net = []
    if isinstance(dilation, int):
        dilations = [dilation] * (len(hidden_sizes) + 1)
    else:
        dilations = dilation
    for i in range(len(sizes) - 1):
        if i != len(sizes) - 2:
            net.append(ResidualConv2d(sizes[i], sizes[i + 1], kernel_size, dilations[i], "circular", float_dtype))
        else:
            net.append(nn.Conv2d(sizes[i], sizes[i + 1], kernel_size,
                                 padding=dilations[i] * (kernel_size - 1) // 2,
                                 stride=1, padding_mode="circular", dilation=dilations[i], dtype=float_dtype, bias=False))
            if use_final_tanh:
                net.append(nn.Tanh())
    return nn.Sequential(*net)

def make_conv_net(*, in_channels, hidden_sizes, out_channels, kernel_size, use_final_tanh, dilation=1,
                  float_dtype=torch.float32):
    """An auxiliary helper function to create a convolutional neural network


    Parameters
    ----------
    in_channels: int
        number of input channels
    hidden_sizes: sequence
        number of channels in the hidden layers. Sets the number of hidden layers.
    out_channels
        number of output channels
    kernel_size: int
        kernel size of each convolutional layer
    use_final_tanh
        whether to use a tanh activation function at the end of the network
    dilation: int or sequence
        dilation of each convolutional layer. If an integer is given then the same dilation is used for all layers.
    float_dtype

    Returns
    -------
        a  convolutional neural network

    """
    sizes = [in_channels] + hidden_sizes + [out_channels]
    net = []
    if isinstance(dilation, int):
        dilations = [dilation] * (len(hidden_sizes) + 1)
    else:
        dilations = dilation
    for i in range(len(sizes) - 1):
        net.append(
            torch.nn.Conv2d(
                sizes[i],
                sizes[i + 1],
                kernel_size,
                padding=dilations[i] * (kernel_size - 1) // 2,
                stride=1,
                padding_mode="circular",
                dilation=dilations[i],
                dtype=float_dtype,
                bias = False
            )
        )
        if i != len(sizes) - 2:
            net.append(torch.nn.LeakyReLU())
        else:
            if use_final_tanh:
                net.append(torch.nn.Tanh())
    return torch.nn.Sequential(*net)


def sample(n_samples, batch_size, prior, layers):
    """Sample from a normalizing flow

    Generates the samples from prior distributions and transforms them using the normalizing flow.
    The samples are generated in batches of size batch_size to reduce memory usage on the GPU.
    Samples are stored in the CPU memory.

    Parameters
    ----------
    n_samples
        number of samples to generate
    batch_size
        number of samples to generate at a time
    prior
        distribution to sample prior configurations from
    layers
        normalizing flow layers
    Returns
    -------
        generated samples and the log probability of the generated samples
    """

    rem_size = n_samples
    samples = []
    log_q = []
    while rem_size > 0:
        with torch.no_grad():
            batch_length = min(rem_size, batch_size)
            x, logq = apply_flow_to_prior(prior, layers, batch_size=batch_length)

        samples.append(x.cpu())
        log_q.append(logq.cpu())

        rem_size -= batch_length

    return torch.cat(samples, 0), torch.cat(log_q, -1)


def calc_dkl(logp, logq):
    """Calculate the Kullback-Leibler divergence D(q|p) between two distributions
    given the log probabilities of samples from each distribution

    Parameters
    ----------
    logp
        logarithm of the target probability of samples
    logq
        logarithm of the proposal probability of samples
    Returns
    -------
        Kullback-Leibler divergence D(q|p)
    """

    return (logq - logp).mean()


def compute_ess_lw(logw):
    """Compute the effective sample size

    Parameters
    ----------
    logw
        log of (unnormalized) importance weights

    Returns
    -------
        effective sample size
    """

    log_ess = 2 * torch.logsumexp(logw, dim=0) - torch.logsumexp(2 * logw, dim=0)
    ess_per_cfg = torch.exp(log_ess) / len(logw)
    return ess_per_cfg


def compute_ess(logp, logq):
    """
    Compute the effective sample size

    Parameters
    ----------
    logp
        logarithm of the target probability of samples
    logq
        logarithm of the proposal probability of samples
    Returns
    -------
        effective sample size
    """

    logw = logp - logq
    return compute_ess_lw(logw)


def calc_action(cfgs, *, action, batch_size, device):
    """Calculate the value of the action of on a  set of configurations

    see the utils.calc_function.calc_function function for more details

    Parameters
    ----------
    cfgs
        configurations on which to evaluate the function
    action
        action to evaluate
    batch_size
        size of the batches to use for the calculation
    device

    kwargs
        additional keyword arguments to pass to the function
    Returns
    -------
        values of the function on the configurations
    """

    return calc_function(cfgs, function=action, batch_size=batch_size, device=device)
