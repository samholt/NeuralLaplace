###########################
# Neural Laplace: Learning diverse classes of differential equations in the Laplace domain
# Author: Samuel Holt
###########################
"""
Laplace Reconstructions
"""
import torch
from torch import Tensor, nn

from .inverse_laplace import CME, DeHoog, Euler, FixedTablot, Fourier, Stehfest
from .transformations import complex_to_spherical_riemann, spherical_to_complex

ILT_ALGORITHMS = {
    "fourier": Fourier,
    "dehoog": DeHoog,
    "cme": CME,
    "euler": Euler,
    "fixed_tablot": FixedTablot,
    "stehfest": Stehfest,
}


def laplace_reconstruct(
    laplace_rep_func,
    p,
    t,
    recon_dim=None,
    ilt_algorithm="fourier",
    use_sphere_projection=True,
    ilt_reconstruction_terms=33,
    options=None,
    compute_deriv=False,
    x0=None,
) -> Tensor:
    r"""Reconstruct trajectories :math:`\mathbf{x}(t)` for a system of Laplace representations.

    Given a parameterized Laplace representation functional :math:`\mathbf{F}(\mathbf{p},\mathbf{s})`,

    .. math::
        \begin{aligned}
            \mathbf{x}(t) & = \text{inverse_laplace_transform}(\mathbf{F}(\mathbf{p},\mathbf{s}), t)
        \end{aligned}

    Where :math:`\mathbf{p}` is a Tensor encoding the initial system state as a latent variable, and :math:`t` is the time points to reconstruct trajectories for.

    The parameterized Laplace representation functional :attr:`laplace_rep_func`, :math:`\mathbf{F}(\mathbf{p},\mathbf{s})` also takes an input complex value :math:`\mathbf{s}`.
    This :math:`\mathbf{s}` is used internally when reconstructing a specified time point with the selected inverse Laplace transform algorithm :attr:`ilt_algorithm`.

    Output dtypes and numerical precision are based on the dtypes of the inputs :attr:`p`.

    The reconstructions :math:`\mathbf{x}(t)` are of shape :math:`(\text{MiniBatchSize}, \text{SeqLen}, d_{\text{obs}})`. Where :math:`\text{SeqLen}` dimension corresponds to the input evaluated time points :math:`t`, and :math:`d_{\text{obs}}` is the trajectory dimension for a given time point, :math:`\mathbf{x}(t)`.

    Args:
        laplace_rep_func (nn.Module): Function that maps an input of two tensors, first a scalar Tensor `p` encoding the initial system state and secondly a complex Tensor `s` used to evaluate the Laplace representation.
        p (Tensor): latent variable Tensor of shape :math:`(\text{MiniBatchSize}, \text{K})`. Where :math:`\text{K}` is a hyperparameter, and can be set by the user to their desired value.
        t (Tensor): time points to reconstruct trajectories for of shape :math:`(\text{MiniBatchSize}, \text{SeqLen})` or :math:`(\text{SeqLen})` if all trajectories use the same time points.
        recon_dim (int): trajectory dimension for a given time point. Corresponds to dim :math:`d_{\text{obs}}`. If not explicitly specified, will use the same last dimension of `p`, i.e. :math:`\text{K}`.
        ilt_algorithm (str): inverse Laplace transform algorithm to use. Default: ``fourier``. Available are {``fourier``, ``dehoog``, ``cme``, ``fixed_tablot``, ``stehfest``}. See inverse_laplace.py for further details.
        use_sphere_projection (bool): this uses the `laplace_rep_func` in the stereographic projection of the Riemann sphere. Default ``True``.
        ilt_reconstruction_terms (int): number of ILT reconstruction terms, i.e. the number of complex :math:`s` points in `laplace_rep_func` to reconstruct a single time point.

    Returns:
        Tensor of reconstructions :math:`\mathbf{x}(t)` of shape :math:`(\text{MiniBatchSize}, \text{SeqLen}, d_{\text{obs}})` or if no mini batch of shape :math:`(\text{SeqLen}, d_{\text{obs}})`. :math:`\text{SeqLen}` dimension corresponds to the different time points :math:`t`. This tensor of reconstructions contains the solved value of :math:`\mathbf{x}` for each desired time point in `t`.

    Raises:
        ValueError: if an invalid `ilt_algorithm` is provided.

    """
    options, recon_dim, p = _check_inputs(
        laplace_rep_func,
        p,
        t,
        recon_dim,
        ilt_algorithm,
        use_sphere_projection,
        ilt_reconstruction_terms,
        options,
    )

    if p.dtype == torch.float64:  # For double precision
        ilt = ILT_ALGORITHMS[ilt_algorithm](
            ilt_reconstruction_terms=ilt_reconstruction_terms,
            torch_float_datatype=torch.double,
            torch_complex_datatype=torch.cdouble,
            **options,
        )
    else:
        ilt = ILT_ALGORITHMS[ilt_algorithm](
            ilt_reconstruction_terms=ilt_reconstruction_terms, **options
        )
    if len(t.shape) == 0:
        time_dim = 1
    elif len(t.shape) == 1:
        time_dim = t.shape[0]
    elif len(t.shape) == 2:
        time_dim = t.shape[1]
    else:
        raise ValueError("Unsupported time tensor shape, please use (batch, time_dim)")
    batch_dim = p.shape[0]
    s, T = ilt.compute_s(torch.squeeze(t))
    T = T
    if use_sphere_projection:
        thetam, phim = complex_to_spherical_riemann(
            torch.flatten(s.real), torch.flatten(s.imag)
        )
        thetam = torch.reshape(thetam, s.real.shape)
        phim = torch.reshape(phim, s.imag.shape)
        sph_coords = torch.cat((thetam, phim), 1)
        s_terms_dim = thetam.shape[1]
        if len(torch.squeeze(t).shape) == 2:
            inputs = torch.cat(
                (
                    sph_coords.view(batch_dim, time_dim, -1),
                    p.view(batch_dim, 1, -1).repeat(1, time_dim, 1),
                ),
                2,
            )
        else:
            inputs = torch.cat(
                (
                    sph_coords.view(1, time_dim, -1).repeat(batch_dim, 1, 1),
                    p.view(batch_dim, 1, -1).repeat(1, time_dim, 1),
                ),
                2,
            )
        theta, phi = laplace_rep_func(inputs)
        sr = spherical_to_complex(theta, phi)
        ss = sr.view(-1, time_dim, recon_dim, s_terms_dim)
    else:
        s_split = torch.cat((s.real, s.imag), 1)
        s_terms_dim = s.shape[1]
        inputs = torch.cat(
            (
                s_split.view(1, time_dim, -1).repeat(batch_dim, 1, 1),
                p.view(batch_dim, 1, -1).repeat(1, time_dim, 1),
            ),
            2,
        )
        s_real, s_imag = laplace_rep_func(inputs)
        so = torch.view_as_complex(
            torch.stack((torch.flatten(s_real), torch.flatten(s_imag)), 1)
        )
        sr = torch.reshape(so, s_real.shape)
        ss = sr.view(-1, time_dim, recon_dim, s_terms_dim)
    if len(torch.squeeze(t).shape) == 2:
        return ilt.line_integrate_all_multi_batch_time(
            ss, t.view(-1, time_dim), T.view(-1, time_dim)
        )
    else:
        return ilt.line_integrate_all_multi(ss, torch.squeeze(t), T)


def _check_inputs(
    laplace_rep_func,
    p,
    t,
    recon_dim,
    ilt_algorithm,
    use_sphere_projection,
    ilt_reconstruction_terms,
    options,
):
    if not isinstance(laplace_rep_func, nn.Module):
        raise RuntimeError("laplace_rep_func must be a descendant of torch.nn.Module")
    if not isinstance(p, Tensor):
        raise RuntimeError("p must be a torch.Tensor type")
    if not isinstance(t, Tensor):
        raise RuntimeError("t must be a torch.Tensor type")
    if not isinstance(use_sphere_projection, bool):
        raise RuntimeError("use_sphere_projection must be a bool type")
    if ilt_algorithm not in ILT_ALGORITHMS:
        raise ValueError(
            'Invalid ILT algorithm "{}". Must be one of {}'.format(
                ilt_algorithm, '{"' + '", "'.join(ILT_ALGORITHMS.keys()) + '"}.'
            )
        )
    if (ilt_reconstruction_terms % 2 == 0) and (ilt_algorithm != "cme"):
        raise ValueError(
            'Invalid "ilt_reconstruction_terms", must be an odd input number. Was given an even number of {}'.format(
                ilt_reconstruction_terms
            )
        )
    if options is None:
        options = {}
    else:
        options = options.copy()
    # if len(p.shape) >= 3:
    # p = torch.squeeze(p)
    if recon_dim is None:
        recon_dim = p.shape[1]
    return options, recon_dim, p
