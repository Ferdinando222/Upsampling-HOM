import torch
import numpy as np
from scipy.special import lpmv
from scipy.special import spherical_jn
import global_variables as gb

def spherical_harmonics(theta, phi, n, m):
    """
    Compute the spherical harmonics Y_nm(theta, phi) for given n and m.

    Args:
        theta (torch.Tensor): Polar angle tensor.
        phi (torch.Tensor): Azimuthal angle tensor.
        m (int): Degree of the spherical harmonics.
        m (int): Order of the spherical harmonics.

    Returns:
        torch.Tensor: Spherical harmonics evaluated at given theta and phi.
    """
    # Associated Legendre polynomial
    P_nm = lpmv(m, n, torch.cos(theta.cpu())).clone().detach().to(gb.device)

    # Normalization factor
    norm = torch.sqrt((2 * n + 1) / (4 * torch.pi) * factorial(n - m) / factorial(n + m))

    # Spherical harmonics
    Y_nm = norm * P_nm * torch.exp(1j * m * phi)
    

    return Y_nm


def spherical_bessel(n, k, r):
    """
    Compute Spherical Bessel Function

    Args:
        n (int): Order Bessel Function
        k (torch.Tensor): Wave number
        r (torch.Tensor): radius

    Returns:
        torch.Tensor: Value of the spherical Bessel function j_n(kr).
    """
    kr = k * r
    return torch.tensor(spherical_jn(n, kr.detach().cpu().numpy())).to(gb.device)

def factorial(n):
    """
    Compute factorial

    Args:
        n (int): Integer number

    Returns:
        int: Factorial of n
    """
    result = torch.tensor(1).to(gb.device)
    for i in range(2, n + 1):
        result *= i
    return result