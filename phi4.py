import numpy as np
import torch

class ScalarPhi4Action:
    def __init__(self, M2, lam) -> None:
        self.M2 = M2
        self.lam = lam
        self.inv_24 = 1.0/24.0

    #def __call__(self, cfgs):
    #    action_density = (0.5*self.M2 +2.0)* cfgs ** 2 + self.inv_24*self.lam * cfgs ** 4
    #    Nd = len(cfgs.shape) - 1

    #    dims = range(1, Nd + 1)
    #    for mu in dims:

    #        action_density -=  cfgs * torch.roll(cfgs, -1, mu)
    #    return torch.sum(action_density, dim=tuple(dims))
    def __call__(self, cfgs):
        # potential term
        action_density = self.M2*cfgs**2 + self.lam*cfgs**4
        # kinetic term (discrete Laplacian)
        Nd = len(cfgs.shape)-1
        dims = range(1,Nd+1)
        for mu in dims:
            action_density += 2*cfgs**2
            action_density -= cfgs*torch.roll(cfgs, -1, mu)
            action_density -= cfgs*torch.roll(cfgs, 1, mu)
        return torch.sum(action_density, dim=tuple(dims))

def free_field_free_energy(L,M2):
    """
    Calculates the free energy of a free scalar field in 2D with mass squared M2 on a LxL lattice.

    :param L: lattice size.
    :param M2: float squared mass
    :return: free field free energy
    """
    q = np.arange(0, L)
    q_0, q_1 = np.meshgrid(q, q)
    k2_0 = 4 * np.sin(np.pi / L * q_0) ** 2
    k2_1 = 4 * np.sin(np.pi / L * q_1) ** 2

    return -0.5 * (L * L) * np.log(2 * np.pi) + 0.5 * np.log(k2_0 + k2_1 + M2).sum()

def mag2(cfgs):
    vol = torch.prod(torch.tensor(cfgs.shape[1:]))
    return torch.mean(
        cfgs.sum(
            dim=tuple(range(1, len(cfgs.shape)))
        )**2/vol
    )

def two_point(phis: torch.Tensor, average: bool = True) -> torch.Tensor:
    """Estimate ``G(x) = <phi(0) phi(x)>``.

    Translational invariance is assumed, so to improve the estimate we compute
        ``mean_y <phi(y) phi(x+y)>``
    using periodic boundary conditions.

    Args:
        phis: Samples of field configurations of shape
            ``(batch size, L_1, ..., L_d)``.
        average: If false, average over samples is not executed.

    Returns:
        Tensor of shape ``(L_1, ..., L_d)`` if ``average`` is true, otherwise
        of shape ``(batch size, L_1, ..., L_d)``.
    """
    corr = torch.fft.fftshift(torch.fft.fftn(phis) * torch.fft.fftn(phis).conj()).real
    return torch.mean(corr, axis=0) if average else corr

def two_point_central(phis: torch.Tensor) -> torch.Tensor:
    """Estimate ``G_c(x) = <phi(0) phi(x)> - <phi(0)> <phi(x)>``.

    Translational invariance is assumed, so to improve the estimate we compute
        ``mean_y <phi(y) phi(x+y)> - <phi(x)> mean_y <phi(x+y)>``
    using periodic boundary conditions.

    Args:
        phis: Samples of field configurations of shape
            ``(batch size, L_1, ..., L_d)``.

    Returns:
        Tensor of shape ``(L_1, ..., L_d)``.
    """
    phis_mean = torch.mean(phis, axis=0)
    outer = phis_mean * torch.mean(phis_mean)

    return two_point(phis, True) - outer

def correlation_length(G: torch.Tensor) -> torch.Tensor:
    """Estimator for the correlation length.

    Args:
        G: Centered two-point function.

    Returns:
        Scalar. Estimate of correlation length.
    """
    Gs = torch.mean(G, axis=0)
    arg = (torch.roll(Gs, 1) + torch.roll(Gs, -1)) / (2 * Gs)
    mp = torch.acosh(arg[1:])
    return 1 / torch.nanmean(mp)
