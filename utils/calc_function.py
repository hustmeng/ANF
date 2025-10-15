import torch


def calc_function(cfgs, *, function, batch_size, device, **kwargs):
    """Calculate the value of a function on a set of configurations

    Given a set of configurations, calculates the value of a function on each of them.
    The function is assumed to be vectorized, i.e. it can take a batch of configurations as input.
    The calculation is done in batches to avoid memory issues on the device. Each batch is  evaluated on the device
    and the results are combined on the CPU.

    Parameters
    ----------
    cfgs
        configurations on which to evaluate the function
    function
        function to evaluate
    batch_size
        size of the batches to use for the calculation
    device

    kwargs
        additional keyword arguments to pass to the function
    Returns
    -------
        values of the function on the configurations
    """
    rem_size = len(cfgs)

    obs = []
    i = 0
    while rem_size > 0:
        with torch.no_grad():
            batch_length = min(rem_size, batch_size)
            o = function(cfgs[i:i + batch_length].to(device), **kwargs)
            obs.append(o.cpu())
            i += batch_length
            rem_size -= batch_length

    return torch.cat(obs, -1)


