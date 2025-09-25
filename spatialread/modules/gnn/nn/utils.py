from typing import Callable, Optional

import torch
from torch import nn as nn

__all__ = ["replicate_module", "derivative_from_atomic", "derivative_from_molecular", "as_dtype"]

from torch.autograd import grad

import importlib
# import torch
from typing import Type, Union, List

TORCH_DTYPES = {
    "float32": torch.float32,
    "float64": torch.float64,
    "float": torch.float,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "half": torch.half,
    "uint8": torch.uint8,
    "int8": torch.int8,
    "int16": torch.int16,
    "short": torch.short,
    "int32": torch.int32,
    "int": torch.int,
    "int64": torch.int64,
    "long": torch.long,
    "complex64": torch.complex64,
    "cfloat": torch.cfloat,
    "complex128": torch.complex128,
    "cdouble": torch.cdouble,
    "quint8": torch.quint8,
    "qint8": torch.qint8,
    "qint32": torch.qint32,
    "bool": torch.bool,
}

TORCH_DTYPES.update({"torch." + k: v for k, v in TORCH_DTYPES.items()})


def as_dtype(dtype_str: str) -> torch.dtype:
    """Convert a string to torch.dtype"""
    return TORCH_DTYPES[dtype_str]


def replicate_module(
    module_factory: Callable[[], nn.Module], n: int, share_params: bool
):
    if share_params:
        module_list = nn.ModuleList([module_factory()] * n)
    else:
        module_list = nn.ModuleList([module_factory() for i in range(n)])
    return module_list


def derivative_from_molecular(
    fx: torch.Tensor,
    dx: torch.Tensor,
    create_graph: bool = False,
    retain_graph: bool = False,
):
    """
    Compute the derivative of `fx` with respect to `dx` if the leading dimension of `fx` is the number of molecules
    (e.g. energies, dipole moments, etc).

    Args:
        fx (torch.Tensor): Tensor for which the derivative is taken.
        dx (torch.Tensor): Derivative.
        create_graph (bool): Create computational graph.
        retain_graph (bool): Keep the computational graph.

    Returns:
        torch.Tensor: derivative of `fx` with respect to `dx`.
    """
    fx_shape = fx.shape
    dx_shape = dx.shape
    # Final shape takes into consideration whether derivative will yield atomic or molecular properties
    final_shape = (dx_shape[0], *fx_shape[1:], *dx_shape[1:])

    fx = fx.view(fx_shape[0], -1)

    dfdx = torch.stack(
        [
            grad(
                fx[..., i],
                dx,
                torch.ones_like(fx[..., i]),
                create_graph=create_graph,
                retain_graph=retain_graph,
            )[0]
            for i in range(fx.shape[1])
        ],
        dim=1,
    )
    dfdx = dfdx.view(final_shape)

    return dfdx


def derivative_from_atomic(
    fx: torch.Tensor,
    dx: torch.Tensor,
    n_atoms: torch.Tensor,
    create_graph: bool = False,
    retain_graph: bool = False,
):
    """
    Compute the derivative of a tensor with the leading dimension of (batch x atoms) with respect to another tensor of
    either dimension (batch * atoms) (e.g. R) or (batch * atom pairs) (e.g. Rij). This function is primarily used for
    computing Hessians and Hessian-like response properties (e.g. nuclear spin-spin couplings). The final tensor will
    have the shape ( batch * atoms * atoms x ....).

    This is quite inefficient, use with care.

    Args:
        fx (torch.Tensor): Tensor for which the derivative is taken.
        dx (torch.Tensor): Derivative.
        n_atoms (torch.Tensor): Tensor containing the number of atoms for each molecule.
        create_graph (bool): Create computational graph.
        retain_graph (bool): Keep the computational graph.

    Returns:
        torch.Tensor: derivative of `fx` with respect to `dx`.
    """
    # Split input tensor for easier bookkeeping
    fxm = fx.split(list(n_atoms))

    dfdx = []

    n_mol = 0
    # Compute all derivatives
    for idx in range(len(fxm)):
        fx = fxm[idx].view(-1)

        # Generate the individual derivatives
        dfdx_mol = []
        for i in range(fx.shape[0]):
            dfdx_i = grad(
                fx[i],
                dx,
                torch.ones_like(fx[i]),
                create_graph=create_graph,
                retain_graph=retain_graph,
            )[0]

            dfdx_mol.append(dfdx_i[n_mol : n_mol + n_atoms[idx], ...])

        # Build molecular matrix and reshape
        dfdx_mol = torch.stack(dfdx_mol, dim=0)
        dfdx_mol = dfdx_mol.view(n_atoms[idx], 3, n_atoms[idx], 3)
        dfdx_mol = dfdx_mol.permute(0, 2, 1, 3)
        dfdx_mol = dfdx_mol.reshape(n_atoms[idx] ** 2, 3, 3)

        dfdx.append(dfdx_mol)

        n_mol += n_atoms[idx]

    # Accumulate everything
    dfdx = torch.cat(dfdx, dim=0)

    return dfdx
