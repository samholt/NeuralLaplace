###########################
# Neural Laplace: Learning diverse classes of differential equations in the Laplace domain
# Author: Samuel Holt
###########################
import sys
from time import time

import numpy as np
import torch
from scipy.special import factorial
from torch import nn

try:
    from ._iltcme import cme_params_factory
except ImportError:
    from _iltcme import cme_params_factory  # For when running this single file as main

EPS = sys.float_info.epsilon
TORCH_FLOAT_DATATYPE = torch.float32
TORCH_COMPLEX_DATATYPE = torch.cfloat

device = "cuda" if torch.cuda.is_available() else "cpu"


def real_vector_to_complex(vector):
    if "ComplexDoubleTensor" not in vector.type():
        if len(vector.shape) == 1:
            return torch.view_as_complex(
                torch.stack((vector, torch.zeros_like(vector)), 1)
            )
        else:
            return torch.reshape(
                torch.view_as_complex(
                    torch.stack(
                        (
                            torch.flatten(vector),
                            torch.zeros_like(torch.flatten(vector)),
                        ),
                        1,
                    )
                ),
                vector.shape,
            )
    else:
        return vector


def complex_numpy_to_complex_torch(sn):
    return torch.view_as_complex(
        torch.stack(
            (
                torch.flatten(torch.Tensor(sn.real)),
                torch.flatten(torch.Tensor(sn.imag)),
            ),
            1,
        )
    )


class InverseLaplaceTransformAlgorithmBase(nn.Module):
    r"""Base class for Inverse Laplace Transform (ILT) Algorithms. This reconstruct trajectories :math:`\mathbf{x}(t)` for a system of Laplace representations.

    Given a parameterized Laplace representation function :math:`\mathbf{F}(\mathbf{s})`,

    .. math::
        \begin{aligned}
            \mathbf{x}(t) & = \text{inverse_laplace_transform}(\mathbf{F}(\mathbf{s}), t)
        \end{aligned}

    Where :math:`t` is the time points to reconstruct trajectories for.

    The Laplace representation function :attr:`fs`, :math:`\mathbf{F}(\mathbf{s})` takes an input complex value :math:`\mathbf{s}`.
    This :math:`\mathbf{s}` is used internally when reconstructing a specified time point with the selected inverse Laplace transform algorithm class.

    The reconstructions :math:`\mathbf{x}(t)` are of shape :math:`(\text{MiniBatchSize}, \text{SeqLen}, d_{\text{obs}})`.
    Where :math:`\text{SeqLen}` dimension corresponds to the input evaluated time points :math:`t`, and :math:`d_{\text{obs}}` is the trajectory dimension for a given time point, :math:`\mathbf{x}(t)`.

    The inverse Laplace transform (ILT) is defined as

    .. math::
        \begin{aligned}
            \hat{\mathbf{x}}(t) = \mathcal{L}^{-1}\{\mathbf{F}(\mathbf{s})\}(t)=\frac{1}{2\pi i} \int_{\sigma - i \infty}^{\sigma + i \infty} \mathbf{F}(\mathbf{s})e^{\mathbf{s}t}d\mathbf{s}
        \end{aligned}

    where the integral refers to the Bromwich contour integral in :math:`\mathbb{C}^d` with the contour :math:`\sigma>0` chosen such that all the singularities of :math:`\mathbf{F}(\mathbf{s})`
    are to the left of it `[1] <https://arxiv.org/abs/2206.04843>`__.

    Many algorithms have been developed to numerically evaluate the ILT Equation (above). On a high level, they involve two steps:

    .. math::
        \begin{aligned}
            \mathcal{Q}(t) &= \text{ILT-Query} (t) \\
            \hat{\mathbf{x}}(t) &= \text{ILT-Compute}\big(\{\mathbf{F}(\mathbf{s})| \mathbf{s} \in \mathcal{Q}(t) \}\big)
        \end{aligned}

    To evaluate :math:`\mathbf{x}(t)` on time points :math:`t \in \mathcal{T} \subset \mathbb{R}`, the algorithms first construct a set of
    `query points` :math:`\mathbf{s} \in \mathcal{Q}(\mathcal{T}) \subset \mathbb{C}`.
    They then compute :math:`\hat{\mathbf{x}}(t)` using the :math:`\mathbf{F}(\mathbf{s})` evaluated on these points.
    The number of query points scales `linearly` with the number of time points, i.e. :math:`|\mathcal{Q}(\mathcal{T})| = b |\mathcal{T}|`, where the constant :math:`b > 1`,
    denotes the number of reconstruction terms per time point and is specific to the algorithm.
    Importantly, the computation complexity of ILT only depends on the `number` of time points, but not their values (e.g. ILT for :math:`t=0` and :math:`t=100` requires the same amount of computation).
    The vast majority of ILT algorithms are differentiable with respect to :math:`\mathbf{F}(\mathbf{s})`, which allows the gradients to be back propagated through the ILT transform `[1] <https://arxiv.org/abs/2206.04843>`__.

    Args:
        ilt_reconstruction_terms (int): number of ILT reconstruction terms, i.e. the number of complex :math:`s` points in `fs` to reconstruct a single time point.
        torch_float_datatype (Torch.dtype): Torch float datatype to use internally in the ILT algorithm, and also output data type of :math:`\mathbf{x}(t)`. Default `torch.float32`.
        torch_complex_datatype (Torch.dtype): Torch complex datatype to use internally in the ILT algorithm. Default `torch.cfloat`.
    """

    def __init__(
        self,
        ilt_reconstruction_terms=33,
        torch_float_datatype=TORCH_FLOAT_DATATYPE,
        torch_complex_datatype=TORCH_COMPLEX_DATATYPE,
    ):
        super(InverseLaplaceTransformAlgorithmBase, self).__init__()
        self.ilt_reconstruction_terms = ilt_reconstruction_terms
        self.torch_float_datatype = torch_float_datatype
        self.torch_complex_datatype = torch_complex_datatype

    def forward(self, fs, ti):
        r"""Reconstructs a trajectory :math:`\mathbf{x}(t)` for a Laplace representation :math:`\mathbf{F}(\mathbf{s})`, at time points :math:`t`.

        Args:
            fs (Torch.nn.Module or Callable): The first parameter.
            ti (Tensor): time points to reconstruct trajectory for of shape :math:`(\text{SeqLen})`.

        Returns:
            Tensor of reconstructions :math:`\mathbf{x}(t)` of shape :math:`(\text{SeqLen}, d_{\text{obs}})`.
            :math:`\text{SeqLen}` dimension corresponds to the different time points :math:`t`.
            This tensor of reconstructions contains the solved value of :math:`\mathbf{x}` for each desired time point in `t`.

        """
        raise NotImplementedError(
            "Forward method is not implemented yet for this ILT algorithm"
        )

    def compute_s(self, ti):
        r"""Computes :math:`\mathbf{s}` to evaluate the Laplace representation :math:`\mathbf{F}(\mathbf{s})` at, from the input time points :math:`t`, using the selected ILT algorithm.

        Args:
            ti (Tensor): time points to reconstruct trajectory for of shape :math:`(\text{SeqLen})`.

        Returns:
            Tensor of complex s points :math:`\mathbf{s}` of shape :math:`(\text{SeqLen}, \text{ReconTerms})`. :math:`\text{SeqLen}` dimension corresponds to the different time points :math:`t`.

        """
        raise NotImplementedError(
            "compute_s method is not implemented yet for this ILT algorithm"
        )

    def line_integrate(self, fp, ti):
        r"""Reconstruct trajectories :math:`\mathbf{x}(t)` for `fp`, Laplace representations evaluated at `s` points from the input `ti` points, :math:`t`, using the selected ILT algorithm.

        Args:
            fp (Tensor): Laplace representation evaluated at `s` points derived from the input time points `ti`. `fp` has shape :math:`(\text{SeqLen}, d_{\text{obs}}, \text{ReconTerms})`
            ti (Tensor): time points to reconstruct trajectory for of shape :math:`(\text{SeqLen})`.

        Returns:
            Tensor of reconstructions :math:`\mathbf{x}(t)` of shape :math:`(\text{SeqLen}, d_{\text{obs}})`. :math:`\text{SeqLen}` dimension corresponds to the different time
            points :math:`t`. This tensor of reconstructions contains the solved value of :math:`\mathbf{x}` for each desired time point in `t`.

        """
        raise NotImplementedError(
            "line_integrate method is not implemented yet for this ILT algorithm"
        )

    def line_integrate_all_multi(self, fp, ti, T):
        r"""Reconstruct trajectories :math:`\mathbf{x}(t)` for `fp`, Laplace representations evaluated at `s` points from the input `ti` points, :math:`t`, using the selected ILT algorithm (takes batch input of `fp`).

        Args:
            fp (Tensor): Laplace representation evaluated at `s` points derived from the input time points `ti`. `fp` has shape :math:`(\text{BatchSize}, \text{SeqLen}, d_{\text{obs}}, \text{ReconTerms})`
            ti (Tensor): time points to reconstruct trajectory for of shape :math:`(\text{SeqLen})`.
            T (Tensor): time points to reconstruct trajectory for used as the reconstruction of times up to `T` time point of shape :math:`(\text{SeqLen})`.
            Best practice for most ILT algorithms is to set `T = ti`, for the ILT algorithms that rely on `T`.

        Returns:
            Tensor of reconstructions :math:`\mathbf{x}(t)` of shape :math:`(\text{BatchSize}, \text{SeqLen}, d_{\text{obs}})`.
            :math:`\text{SeqLen}` dimension corresponds to the different time points :math:`t`. This tensor of reconstructions contains the solved value of :math:`\mathbf{x}` for each desired time point in `t`.

        """
        # Input shapes
        # fp [batch, times, data_dims, approximation_terms * 2 + 1]
        # ti [times]
        # T [times]
        # Returns
        # [batch, times, data_dims]
        raise NotImplementedError(
            "line_integrate_all_multi method is not implemented yet for this ILT algorithm"
        )


class FixedTablot(InverseLaplaceTransformAlgorithmBase):
    r"""Inherits from :meth:`torchlaplace.inverse_laplace.InverseLaplaceTransformAlgorithmBase`.

    Reconstruct trajectories :math:`\mathbf{x}(t)` for a system of Laplace representations.
    Given a parameterized Laplace representation function :math:`\mathbf{F}(\mathbf{s})`.

    Deforms the Bromwich contour around the negative real axis, where :math:`\mathbf{F}(\mathbf{s})` must not overflow as :math:`\mathbf{s} \to -\infty`,
    and makes the Bromwich contour integral rapidly converge as :math:`\mathbf{s} \to -\infty` causes :math:`e^{\mathbf{s}t} \to 0` in ILT Equation.
    We implemented the Fixed Tablot method, which is simple to implement.
    However it suffers from not being able to model solutions that have large sinusoidal components and instead is optimized for modelling decaying exponential solutions.
    We note that whilst it can approximate some small sinusoidal components, for an adaptive time contour, the sinusoidal components that can be represented decrease when
    modelling longer time trajectories, and in the limit for long time horizons, allow only representations of decaying exponentials.

    .. note::
        References: Abate, J. and Valko, P. P. Multi-precision laplace transform inversion. International Journal for Numerical Methods in Engineering, 60:979–993, 2004.

        Talbot, A. The accurate numerical inversion of laplace transforms. IMA Journal of Applied Mathematics, 23(1): 97–120, 1979.

    Args:
        ilt_reconstruction_terms (int): number of ILT reconstruction terms, i.e. the number of complex :math:`s` points in `fs` to reconstruct a single time point.
        torch_float_datatype (Torch.dtype): Torch float datatype to use internally in the ILT algorithm, and also output data type of :math:`\mathbf{x}(t)`. Default `torch.float32`.
        torch_complex_datatype (Torch.dtype): Torch complex datatype to use internally in the ILT algorithm. Default `torch.cfloat`.
    """

    def __init__(
        self,
        ilt_reconstruction_terms=33,
        torch_float_datatype=TORCH_FLOAT_DATATYPE,
        torch_complex_datatype=TORCH_COMPLEX_DATATYPE,
    ):
        super().__init__(
            ilt_reconstruction_terms, torch_float_datatype, torch_complex_datatype
        )
        M = int((ilt_reconstruction_terms - 1) / 2)
        self.M = M
        self.rdt = torch.Tensor([(2.0 / 5.0) * self.M]).to(device)
        k = torch.arange(1, M, dtype=torch_float_datatype, device=torch.device(device))
        self.theta = (k * torch.pi) / M
        self.sdt = self.rdt * self.theta * (1 / torch.tan(self.theta) + 1j)
        self.torch_float_datatype = torch_float_datatype

    def forward(self, fs, ti, time_max=None):
        r"""Reconstructs a trajectory :math:`\mathbf{x}(t)` for a Laplace representation :math:`\mathbf{F}(\mathbf{s})`, at time points :math:`t`.

        Args:
            fs (Torch.nn.Module or Callable): The first parameter.
            ti (Tensor): time points to reconstruct trajectory for of shape :math:`(\text{SeqLen})`.
            time_max (float): Maximum time to compute reconstruction up to. Best results use default of None, as this uses `time_max=ti`.

        Returns:
            Tensor of reconstructions :math:`\mathbf{x}(t)` of shape :math:`(\text{SeqLen}, d_{\text{obs}})`. :math:`\text{SeqLen}` dimension corresponds to the different time
            points :math:`t`. This tensor of reconstructions contains the solved value of :math:`\mathbf{x}` for each desired time point in `t`.

        """
        # fs: C^1 -> C^1
        t = real_vector_to_complex(ti)
        if time_max is not None:
            self.t_time_max = (
                real_vector_to_complex(
                    torch.ones(
                        t.shape,
                        dtype=self.torch_float_datatype,
                        device=torch.device(device),
                    )
                )
                * time_max
            )
            s = torch.matmul(1 / self.t_time_max.view(-1, 1), self.sdt.view(1, -1))
            f0 = (
                torch.exp((self.rdt * t) / time_max)
                * fs(self.rdt / self.t_time_max)
                / 2.0
            )
            fp = torch.reshape(fs(torch.flatten(s)), s.shape)
            pans = (
                torch.exp((self.rdt * t) / time_max).view(-1, 1)
                * fp
                * (
                    1.0
                    + 1j * self.theta * (1.0 + (1 / torch.tan(self.theta) ** 2))
                    - 1j * (1 / torch.tan(self.theta))
                )
            )
            result = (2.0 / 5.0) * (torch.sum(pans, 1) + f0) * (1 / time_max)
            return result.real
        else:
            s = torch.matmul(1 / t.view(-1, 1), self.sdt.view(1, -1))
            f0 = torch.exp(self.rdt) * fs(self.rdt / t) / 2.0
            fp = torch.reshape(fs(torch.flatten(s)), s.shape)
            pans = (
                torch.exp(self.sdt)
                * fp
                * (
                    1.0
                    + 1j * self.theta * (1.0 + (1 / torch.tan(self.theta) ** 2))
                    - 1j * (1 / torch.tan(self.theta))
                )
            )
            result = (2.0 / 5.0) * (torch.sum(pans, 1) + f0) * (1 / t)
            return result.real

    def compute_s(self, ti, time_max=None):
        r"""Computes :math:`\mathbf{s}` to evaluate the Laplace representation :math:`\mathbf{F}(\mathbf{s})` at, from the input time points :math:`t`, using the selected ILT algorithm.

        Args:
            ti (Tensor): time points to reconstruct trajectory for of shape :math:`(\text{SeqLen})`.
            time_max (float): Maximum time to compute reconstruction up to. Best results use default of None, as this uses `time_max=ti`.

        Returns:
            Tensor of complex s points :math:`\mathbf{s}` of shape :math:`(\text{SeqLen}, \text{ReconTerms})`. :math:`\text{SeqLen}` dimension corresponds to the different time points :math:`t`.

        """
        # Split of forward into generating s values, then taking function evaluations at s points and integrating along that line
        t = real_vector_to_complex(ti)
        if time_max is not None:
            self.t_time_max = (
                real_vector_to_complex(
                    torch.ones(
                        t.shape,
                        dtype=self.torch_float_datatype,
                        device=torch.device(device),
                    )
                )
                * time_max
            )
            s = torch.matmul(1 / self.t_time_max.view(-1, 1), self.sdt.view(1, -1))
            s0 = self.rdt / self.t_time_max
            return torch.hstack((s0.view(-1, 1), s)), self.rdt / time_max
        else:
            s = torch.matmul(1 / t.view(-1, 1), self.sdt.view(1, -1))
            s0 = self.rdt / t
            return torch.hstack((s0.view(-1, 1), s)), self.rdt / t

    def line_integrate(self, fp, ti, time_max=None):
        r"""Reconstruct trajectories :math:`\mathbf{x}(t)` for `fp`, Laplace representations evaluated at `s` points from the input `ti` points, :math:`t`, using the selected ILT algorithm.

        Args:
            fp (Tensor): Laplace representation evaluated at `s` points derived from the input time points `ti`. `fp` has shape :math:`(\text{SeqLen}, d_{\text{obs}}, \text{ReconTerms})`
            ti (Tensor): time points to reconstruct trajectory for of shape :math:`(\text{SeqLen})`.
            time_max (float): Maximum time to compute reconstruction up to. Best results use default of None, as this uses `time_max=ti`.

        Returns:
            Tensor of reconstructions :math:`\mathbf{x}(t)` of shape :math:`(\text{SeqLen}, d_{\text{obs}})`. :math:`\text{SeqLen}` dimension corresponds to the different time
            points :math:`t`. This tensor of reconstructions contains the solved value of :math:`\mathbf{x}` for each desired time point in `t`.

        """
        t = real_vector_to_complex(ti)
        if time_max is not None:
            f0 = torch.exp((self.rdt * t) / time_max) * fp[:, 0] / 2.0
            pans = (
                torch.exp((self.rdt * t) / time_max).view(-1, 1)
                * fp[:, 1:]
                * (
                    1.0
                    + 1j * self.theta * (1.0 + (1 / torch.tan(self.theta) ** 2))
                    - 1j * (1 / torch.tan(self.theta))
                )
            )
            result = (2.0 / 5.0) * (torch.sum(pans, 1) + f0) * (1 / time_max)
            return result.real
        else:
            f0 = torch.exp(self.rdt) * fp[:, 0] / 2.0
            pans = (
                torch.exp(self.sdt)
                * fp[:, 1:]
                * (
                    1.0
                    + 1j * self.theta * (1.0 + (1 / torch.tan(self.theta) ** 2))
                    - 1j * (1 / torch.tan(self.theta))
                )
            )
            result = (2.0 / 5.0) * (torch.sum(pans, 1) + f0) / t
            return result.real

    def line_integrate_multi(self, fp, ti, time_max=None):
        r"""Reconstruct trajectories :math:`\mathbf{x}(t)` for `fp`, Laplace representations evaluated at `s` points from the input `ti` points, :math:`t`, using the selected ILT algorithm.

        Args:
            fp (Tensor): Laplace representation evaluated at `s` points derived from the input time points `ti`. `fp` has shape :math:`(\text{SeqLen}, d_{\text{obs}}, \text{ReconTerms})`
            ti (Tensor): time points to reconstruct trajectory for of shape :math:`(\text{SeqLen})`.
            time_max (float): Maximum time to compute reconstruction up to. Best results use default of None, as this uses `time_max=ti`.

        Returns:
            Tensor of reconstructions :math:`\mathbf{x}(t)` of shape :math:`(\text{SeqLen}, d_{\text{obs}})`. :math:`\text{SeqLen}` dimension corresponds to the different time points
            :math:`t`. This tensor of reconstructions contains the solved value of :math:`\mathbf{x}` for each desired time point in `t`.

        """
        t = real_vector_to_complex(ti)
        if time_max is not None:
            f0 = torch.exp((self.rdt * t) / time_max) * fp[:, :, 0] / 2.0
            pans = (
                torch.exp((self.rdt * t) / time_max).view(-1, 1)
                * fp[:, :, 1:]
                * (
                    1.0
                    + 1j * self.theta * (1.0 + (1 / torch.tan(self.theta) ** 2))
                    - 1j * (1 / torch.tan(self.theta))
                )
            )
            result = (2.0 / 5.0) * (torch.sum(pans, 2) + f0) * (1 / time_max)
            return result.real
        else:
            f0 = torch.exp(self.rdt) * fp[:, :, 0] / 2.0
            pans = (
                torch.exp(self.sdt)
                * fp[:, :, 1:]
                * (
                    1.0
                    + 1j * self.theta * (1.0 + (1 / torch.tan(self.theta) ** 2))
                    - 1j * (1 / torch.tan(self.theta))
                )
            )
            result = (2.0 / 5.0) * (torch.sum(pans, 2) + f0) / t
            return result.real


class Stehfest(InverseLaplaceTransformAlgorithmBase):
    r"""Inherits from :meth:`torchlaplace.inverse_laplace.InverseLaplaceTransformAlgorithmBase`.

    Reconstruct trajectories :math:`\mathbf{x}(t)` for a system of Laplace representations.
    Given a parameterized Laplace representation function :math:`\mathbf{F}(\mathbf{s})`.

    Uses a discrete version of the Post-Widder formula that is an approximation for ILT Equation using a power series expansion of real part of :math:`\mathbf{s}`.
    It has internal terms that alternate in sign and become large as the order of approximation is increased, and suffers from numerical precision issues for large orders of approximation.
    It is fairly easy to implement.

    .. note::
        Reference: Al-Shuaibi, A. Inversion of the laplace transform via post—widder formula. Integral Transforms and Special Functions, 11(3):225–232, 2001.

    Args:
        ilt_reconstruction_terms (int): number of ILT reconstruction terms, i.e. the number of complex :math:`s` points in `fs` to reconstruct a single time point.
        torch_float_datatype (Torch.dtype): Torch float datatype to use internally in the ILT algorithm, and also output data type of :math:`\mathbf{x}(t)`. Default `torch.float32`.
        torch_complex_datatype (Torch.dtype): Torch complex datatype to use internally in the ILT algorithm. Default `torch.cfloat`.
    """

    def __init__(
        self,
        ilt_reconstruction_terms=33,
        torch_float_datatype=TORCH_FLOAT_DATATYPE,
        torch_complex_datatype=TORCH_COMPLEX_DATATYPE,
    ):
        # Also computes each s for each time evaluated at
        super().__init__(
            ilt_reconstruction_terms, torch_float_datatype, torch_complex_datatype
        )
        self.torch_complex_datatype = torch_complex_datatype
        M = int((ilt_reconstruction_terms - 1) / 2)
        self.M = M
        self.ln2 = torch.log(torch.Tensor([2.0]).to(device))
        # _coeff routine requires even degree
        if M % 2 > 0:
            self.M += 1

        self.V = self._coeff()
        p_real_dt = (
            torch.arange(
                1, M + 1, dtype=torch_float_datatype, device=torch.device(device)
            )
            * self.ln2
        )
        # NB: s is real
        self.sdt = real_vector_to_complex(p_real_dt)

    def _coeff(self):
        r"""Salzer summation weights (aka, "Stehfest coefficients")
        only depend on the approximation order (M) and the precision"""

        M2 = int(self.M / 2.0)  # checked earlier that M is even

        V = torch.empty(
            (self.M,), dtype=self.torch_complex_datatype, device=torch.device(device)
        )

        def fac(x):
            return float(factorial(x, exact=True))

        # Salzer summation weights
        # get very large in magnitude and oscillate in sign,
        # if the precision is not high enough, there will be
        # catastrophic cancellation
        for k in range(1, self.M + 1):
            z = torch.zeros(
                (min(k, M2) + 1,),
                dtype=self.torch_complex_datatype,
                device=torch.device(device),
            )
            for j in range(int((k + 1) / 2.0), min(k, M2) + 1):
                z[j] = (
                    j**M2
                    * fac(2 * j)
                    / (fac(M2 - j) * fac(j) * fac(j - 1) * fac(k - j) * fac(2 * j - k))
                )
            V[k - 1] = (-1) ** (k + M2) * torch.sum(z)

        return V

    def compute_s(self, ti):
        t = real_vector_to_complex(ti)
        return torch.matmul((1 / t.view(-1, 1)), self.sdt.view(1, -1))

    def line_integrate(self, fp, ti):
        t = real_vector_to_complex(ti)
        result = torch.matmul(fp, self.V) * self.ln2 / t
        # ignore any small imaginary part
        return result.real

    def forward(self, fs, ti):
        # fs: C^1 -> C^1
        t = real_vector_to_complex(ti)
        s = torch.matmul((1 / t.view(-1, 1)), self.sdt.view(1, -1))
        fp = torch.reshape(fs(torch.flatten(s)), s.shape)
        result = torch.matmul(fp, self.V) * self.ln2 / t
        # ignore any small imaginary part
        return result.real


class DeHoog(InverseLaplaceTransformAlgorithmBase):
    r"""Inherits from :meth:`torchlaplace.inverse_laplace.InverseLaplaceTransformAlgorithmBase`.

    Reconstruct trajectories :math:`\mathbf{x}(t)` for a system of Laplace representations.
    Given a parameterized Laplace representation function :math:`\mathbf{F}(\mathbf{s})`.

    `De Hoog` Is an accelerated version of the Fouier ILT, defined in :meth:`torchlaplace.inverse_laplace.Fourier`.
    It uses a non-linear double acceleration, using Padé approximation along with a remainder term for the series.
    This is somewhat complicated to implement, due to the recurrence operations to represent the Padé approximation, due to this although higher precision, the gradients
    have to propagate through many recurrence relation paths, making it slow to use in practice compared to Fourier (FSI), however more accurate when we can afford the additional time complexity.

    .. note::
        Reference: De Hoog, F. R., Knight, J., and Stokes, A. An improved method for numerical inversion of laplace transforms.
        SIAM Journal on Scientific and Statistical Computing, 3(3):357–366, 1982

    Args:
        ilt_reconstruction_terms (int): number of ILT reconstruction terms, i.e. the number of complex :math:`s` points in `fs` to reconstruct a single time point.
        torch_float_datatype (Torch.dtype): Torch float datatype to use internally in the ILT algorithm, and also output data type of :math:`\mathbf{x}(t)`. Default `torch.float32`.
        torch_complex_datatype (Torch.dtype): Torch complex datatype to use internally in the ILT algorithm. Default `torch.cfloat`.
        alpha (float): :math:`\alpha`, default :math:`\alpha=10e-10`.
        tol (float): desired tolerance, if not specified simply related to alpha as :math:`\text{tol}=10\alpha`.
        scale (float): scaling factor (tuneable), default 2.0.
    """

    def __init__(
        self,
        ilt_reconstruction_terms=33,
        alpha=1.0e-10,
        tol=None,
        scale=2.0,
        torch_float_datatype=TORCH_FLOAT_DATATYPE,
        torch_complex_datatype=TORCH_COMPLEX_DATATYPE,
    ):
        super().__init__(
            ilt_reconstruction_terms, torch_float_datatype, torch_complex_datatype
        )
        M = int((ilt_reconstruction_terms - 1) / 2)
        self.M = M
        self.alpha = torch.Tensor([alpha]).to(device)
        # desired tolerance (here simply related to alpha)
        if tol is not None:
            self.tol = torch.Tensor([tol]).to(device)
        else:
            self.tol = self.alpha * 10.0
        self.nt = ilt_reconstruction_terms  # number of terms in approximation
        # scaling factor (likely tune-able, but 2 is typical)
        self.scale = torch.Tensor([scale]).to(device)
        self.torch_float_datatype = torch_float_datatype
        self.torch_complex_datatype = torch_complex_datatype

    def compute_fixed_s(self, time_max):
        r"""Computes :math:`\mathbf{s}` to evaluate the Laplace representation :math:`\mathbf{F}(\mathbf{s})` at for a single input time point `time_max`, using the selected ILT algorithm.

        Args:
            time_max (float): Maximum time to compute reconstruction up to, a single time point to generate `s` for.

        Returns:
            Tensor of complex s points :math:`\mathbf{s}` of shape :math:`(\text{ReconTerms})`.

        """
        NT = self.nt
        tmax = torch.Tensor([time_max]).to(device)
        T = torch.Tensor([self.scale * tmax]).to(device)

        gamma = self.alpha - torch.log(self.tol) / (self.scale * T)
        s = (
            gamma
            + 1j
            * torch.pi
            * torch.arange(
                NT, dtype=self.torch_float_datatype, device=torch.device(device)
            )
            / T
        )
        return s

    def fixed_line_integrate(self, fp, ti, time_max):
        r"""Reconstruct trajectories :math:`\mathbf{x}(t)` for `fp`, Laplace representations evaluated at `s` points from the input `ti` points, :math:`t`, using the selected ILT algorithm.

        Args:
            fp (Tensor): Laplace representation evaluated at `s` points derived from the input time points `ti`. `fp` has shape :math:`(\text{SeqLen}, d_{\text{obs}}, \text{ReconTerms})`
            ti (Tensor): time points to reconstruct trajectory for of shape :math:`(\text{SeqLen})`.
            T (Tensor): time points to reconstruct trajectory for used as the reconstruction of times up to `T` time point of shape :math:`(\text{SeqLen})`. Best practice for most ILT
            algorithms is to set `T = ti`, for the ILT algorithms that rely on `T`.
            time_max (float): Maximum time to compute reconstruction up to, a single time point to generate `s` for.

        Returns:
            Tensor of reconstructions :math:`\mathbf{x}(t)` of shape :math:`(\text{SeqLen}, d_{\text{obs}})`. :math:`\text{SeqLen}` dimension corresponds to the different time points
            :math:`t`. This tensor of reconstructions contains the solved value of :math:`\mathbf{x}` for each desired time point in `t`.

        """
        # fs: C^1 -> C^1
        tv = real_vector_to_complex(ti)
        M = self.M
        NT = self.nt
        tmax = torch.Tensor([time_max]).to(device)
        T = torch.Tensor([self.scale * tmax]).to(device)
        gamma = self.alpha - torch.log(self.tol) / (self.scale * T)

        # would it be useful to try re-using
        # space between e&q ?
        e = torch.empty(
            (NT, M + 1), dtype=self.torch_complex_datatype, device=torch.device(device)
        )
        q = torch.empty(
            (2 * M, M), dtype=self.torch_complex_datatype, device=torch.device(device)
        )
        d = torch.empty(
            (NT,), dtype=self.torch_complex_datatype, device=torch.device(device)
        )

        es = []
        qs = []

        # initialize Q-D table
        q = fp[1 : 2 * M + 1] / fp[0 : 2 * M]
        q[0] = fp[1] / (fp[0] / 2.0)
        qs.append(q)

        # rhombus rule for filling triangular Q-D table (e & q)
        for r in range(1, M + 1):
            # start with e, column 1, 0:2*M-2
            mr = 2 * (M - r) + 1
            if r == 1:
                e = torch.zeros(
                    (NT,),
                    dtype=self.torch_complex_datatype,
                    device=torch.device(device),
                )
                e[0:mr] = qs[r - 1][1 : mr + 1] - qs[r - 1][0:mr]
                es.append(e)
            else:
                e = torch.zeros(
                    (NT,),
                    dtype=self.torch_complex_datatype,
                    device=torch.device(device),
                )
                e[0:mr] = (
                    qs[r - 1][1 : mr + 1] - qs[r - 1][0:mr] + es[r - 2][1 : mr + 1]
                )
                es.append(e)
            if not r == M:
                q = torch.zeros(
                    (2 * M,),
                    dtype=self.torch_complex_datatype,
                    device=torch.device(device),
                )
                q[0:mr] = (
                    qs[r - 1][1 : mr + 1] * es[r - 1][1 : mr + 1] / es[r - 1][0:mr]
                )
                qs.append(q)

        # build up continued fraction coefficients (d)
        d[0] = fp[0] / 2.0
        for r in range(1, M + 1):
            d[2 * r - 1] = -qs[r - 1][0]  # even terms
            d[2 * r] = -es[r - 1][0]  # odd terms

        # seed A and B for recurrence
        Aim1 = torch.view_as_complex(torch.Tensor([0, 0]).to(device))
        Ai = d[0]
        Bim1 = torch.view_as_complex(torch.Tensor([1, 0]).to(device))
        Bi = torch.view_as_complex(torch.Tensor([1, 0]).to(device))

        # base of the power series
        z = torch.exp(1j * torch.pi * ti / T)

        # coefficients of Pade approximation (A & B)
        # using recurrence for all but last term
        for i in range(1, 2 * M):
            Ait = Ai + d[i] * Aim1 * z
            Aim1 = Ai
            Ai = Ait
            Bit = Bi + d[i] * Bim1 * z
            Bim1 = Bi
            Bi = Bit

        # "improved remainder" to continued fraction
        brem = (1.0 + (d[2 * M - 1] - d[2 * M]) * z) / 2.0
        rem = -brem * (1.0 - torch.sqrt(1.0 + d[2 * M] * z / brem**2))

        # last term of recurrence using new remainder
        A_2mp1 = Ai + rem * Aim1
        B_2mp1 = Bi + rem * Bim1

        # diagonal Pade approximation
        # F=A/B represents accelerated trapezoid rule
        return (torch.exp(gamma * tv) / T * (A_2mp1 / B_2mp1).real).real

    def compute_s(self, ti, time_max=None, Ti=None):
        r"""Computes :math:`\mathbf{s}` to evaluate the Laplace representation :math:`\mathbf{F}(\mathbf{s})` at, from the input time points :math:`t`, using the selected ILT algorithm.

        Args:
            ti (Tensor): time points to reconstruct trajectory for of shape :math:`(\text{SeqLen})`.
            time_max (float): Maximum time to compute reconstruction up to. Best results use default of None, as this uses `time_max=ti`.
            Ti (float): Scaled maximum time to compute reconstruction up to. Best results use default of None, as this uses `T=time_max * self.scale`.

        Returns:
            Tensor of complex s points :math:`\mathbf{s}` of shape :math:`(\text{SeqLen}, \text{ReconTerms})`. :math:`\text{SeqLen}` dimension corresponds to the different time points :math:`t`.

        """
        tv = real_vector_to_complex(ti)
        NT = self.nt
        if time_max is not None:
            tmax = torch.Tensor([time_max]).to(device) * torch.ones_like(tv)
        else:
            tmax = tv
        if Ti is not None:
            if type(Ti) != self.torch_float_datatype:
                raise ValueError("Invalid Ti type")
            T = Ti
        else:
            T = (tmax * self.scale).to(self.torch_complex_datatype)
        gamma = self.alpha - torch.log(self.tol) / (self.scale * T)
        si = torch.matmul(
            1 / T.view(-1, 1),
            1j
            * torch.pi
            * torch.arange(
                NT, dtype=self.torch_float_datatype, device=torch.device(device)
            ).view(1, -1),
        )
        return gamma.view(-1, 1) + si, T

    def line_integrate(self, fp, ti, T):
        r"""Reconstruct trajectories :math:`\mathbf{x}(t)` for `fp`, Laplace representations evaluated at `s` points from the input `ti` points, :math:`t`, using the selected ILT algorithm.

        Args:
            fp (Tensor): Laplace representation evaluated at `s` points derived from the input time points `ti`. `fp` has shape :math:`(\text{SeqLen}, d_{\text{obs}}, \text{ReconTerms})`
            ti (Tensor): time points to reconstruct trajectory for of shape :math:`(\text{SeqLen})`.
            T (Tensor): time points to reconstruct trajectory for used as the reconstruction of times up to `T` time point of shape :math:`(\text{SeqLen})`. Best practice for most ILT
            algorithms is to set `T = ti`, for the ILT algorithms that rely on `T`.

        Returns:
            Tensor of reconstructions :math:`\mathbf{x}(t)` of shape :math:`(\text{SeqLen}, d_{\text{obs}})`. :math:`\text{SeqLen}` dimension corresponds to the different time points
            :math:`t`. This tensor of reconstructions contains the solved value of :math:`\mathbf{x}` for each desired time point in `t`.

        """
        t = real_vector_to_complex(ti)
        M = self.M
        NT = self.nt
        t_samples = t.shape[0]
        gamma = self.alpha - torch.log(self.tol) / (self.scale * T)

        # would it be useful to try re-using
        # space between e&q ?
        e = torch.empty(
            (t_samples, NT, M + 1),
            dtype=self.torch_complex_datatype,
            device=torch.device(device),
        )
        q = torch.empty(
            (t_samples, 2 * M, M),
            dtype=self.torch_complex_datatype,
            device=torch.device(device),
        )
        d = torch.empty(
            (
                t_samples,
                NT,
            ),
            dtype=self.torch_complex_datatype,
            device=torch.device(device),
        )

        es = []
        qs = []

        # initialize Q-D table
        q = fp[:, 1 : 2 * M + 1] / fp[:, 0 : 2 * M]
        q[:, 0] = fp[:, 1] / (fp[:, 0] / 2.0)
        qs.append(q)

        # rhombus rule for filling triangular Q-D table (e & q)
        for r in range(1, M + 1):
            # start with e, column 1, 0:2*M-2
            mr = 2 * (M - r) + 1
            if r == 1:
                e = torch.zeros(
                    (
                        t_samples,
                        NT,
                    ),
                    dtype=self.torch_complex_datatype,
                    device=torch.device(device),
                )
                e[:, 0:mr] = qs[r - 1][:, 1 : mr + 1] - qs[r - 1][:, 0:mr]
                es.append(e)
            else:
                e = torch.zeros(
                    (
                        t_samples,
                        NT,
                    ),
                    dtype=self.torch_complex_datatype,
                    device=torch.device(device),
                )
                e[:, 0:mr] = (
                    qs[r - 1][:, 1 : mr + 1]
                    - qs[r - 1][:, 0:mr]
                    + es[r - 2][:, 1 : mr + 1]
                )
                es.append(e)
            if not r == M:
                q = torch.zeros(
                    (
                        t_samples,
                        2 * M,
                    ),
                    dtype=self.torch_complex_datatype,
                    device=torch.device(device),
                )
                q[:, 0:mr] = (
                    qs[r - 1][:, 1 : mr + 1]
                    * es[r - 1][:, 1 : mr + 1]
                    / es[r - 1][:, 0:mr]
                )
                qs.append(q)

        # build up continued fraction coefficients (d)
        d[:, 0] = fp[:, 0] / 2.0
        for r in range(1, M + 1):
            d[:, 2 * r - 1] = -qs[r - 1][:, 0]  # even terms
            d[:, 2 * r] = -es[r - 1][:, 0]  # odd terms

        # seed A and B for recurrence
        Aim1 = torch.view_as_complex(torch.Tensor([0, 0]).to(device))
        Ai = d[:, 0]
        Bim1 = torch.view_as_complex(torch.Tensor([1, 0]).to(device))
        Bi = torch.view_as_complex(torch.Tensor([1, 0]).to(device))

        # base of the power series
        z = torch.exp(1j * torch.pi * t / T)

        # coefficients of Pade approximation (A & B)
        # using recurrence for all but last term
        for i in range(1, 2 * M):
            Ait = Ai + d[:, i] * Aim1 * z
            Aim1 = Ai
            Ai = Ait
            Bit = Bi + d[:, i] * Bim1 * z
            Bim1 = Bi
            Bi = Bit

        # "improved remainder" to continued fraction
        brem = (1.0 + (d[:, 2 * M - 1] - d[:, 2 * M]) * z) / 2.0
        rem = -brem * (1.0 - torch.sqrt(1.0 + d[:, 2 * M] * z / brem**2))

        # last term of recurrence using new remainder
        A_2mp1 = Ai + rem * Aim1
        B_2mp1 = Bi + rem * Bim1

        # diagonal Pade approximation
        # F=A/B represents accelerated trapezoid rule
        result = torch.exp(gamma * t) / T * (A_2mp1 / B_2mp1).real

        return result.real

    def line_integrate_all_multi(self, fp, ti, T):
        r"""Reconstruct trajectories :math:`\mathbf{x}(t)` for `fp`, Laplace representations evaluated at `s` points from the input `ti` points, :math:`t`, using the selected ILT algorithm (takes batch input of `fp`).

        Args:
            fp (Tensor): Laplace representation evaluated at `s` points derived from the input time points `ti`. `fp` has shape :math:`(\text{BatchSize}, \text{SeqLen}, d_{\text{obs}}, \text{ReconTerms})`
            ti (Tensor): time points to reconstruct trajectory for of shape :math:`(\text{SeqLen})`.
            T (Tensor): time points to reconstruct trajectory for used as the reconstruction of times up to `T` time point of shape :math:`(\text{SeqLen})`. Best practice for most ILT
            algorithms is to set `T = ti`, for the ILT algorithms that rely on `T`.

        Returns:
            Tensor of reconstructions :math:`\mathbf{x}(t)` of shape :math:`(\text{BatchSize}, \text{SeqLen}, d_{\text{obs}})`. :math:`\text{SeqLen}` dimension corresponds to the different
            time points :math:`t`. This tensor of reconstructions contains the solved value of :math:`\mathbf{x}` for each desired time point in `t`.

        """
        # Input shapes
        # fp [batch, times, data_dims, approximation_terms * 2 + 1]
        # ti [times]
        # T [times]
        # Returns
        # [batch, times, data_dims]
        samples = []
        for traj in fp:
            data_dims = []
            for data_dim_idx in range(fp.shape[2]):
                data_dims.append(self.line_integrate(traj[:, data_dim_idx, :], ti, T))
            samples.append(torch.stack(data_dims, 1))
        return torch.stack(samples)

    def forward(self, fs, ti, time_max=None, Ti=None):
        r"""Reconstructs a trajectory :math:`\mathbf{x}(t)` for a Laplace representation :math:`\mathbf{F}(\mathbf{s})`, at time points :math:`t`.

        Args:
            fs (Torch.nn.Module or Callable): The first parameter.
            ti (Tensor): time points to reconstruct trajectory for of shape :math:`(\text{SeqLen})`.
            time_max (float): Maximum time to compute reconstruction up to. Best results use default of None, as this uses `time_max=ti`.
            Ti (float): Scaled maximum time to compute reconstruction up to. Best results use default of None, as this uses `T=time_max * self.scale`.

        Returns:
            Tensor of reconstructions :math:`\mathbf{x}(t)` of shape :math:`(\text{SeqLen}, d_{\text{obs}})`. :math:`\text{SeqLen}` dimension corresponds to the different time points
            :math:`t`. This tensor of reconstructions contains the solved value of :math:`\mathbf{x}` for each desired time point in `t`.

        """
        # fs: C^1 -> C^1
        t = real_vector_to_complex(ti)
        M = self.M
        NT = self.nt
        t_samples = t.shape[0]

        if time_max is not None:
            tmax = torch.Tensor([time_max]).to(device) * torch.ones_like(t)
        else:
            tmax = t
        if Ti is not None:
            if type(Ti) != self.torch_float_datatype:
                raise ValueError("Invalid Ti types")
            T = Ti
        else:
            T = (tmax * self.scale).to(self.torch_complex_datatype)
        gamma = self.alpha - torch.log(self.tol) / (self.scale * T)
        si = torch.matmul(
            1 / T.view(-1, 1),
            1j
            * torch.pi
            * torch.arange(
                NT, dtype=self.torch_float_datatype, device=torch.device(device)
            ).view(1, -1),
        )
        fp = fs(gamma.view(-1, 1) + si)

        # would it be useful to try re-using
        # space between e&q ?
        e = torch.empty(
            (t_samples, NT, M + 1),
            dtype=self.torch_complex_datatype,
            device=torch.device(device),
        )
        q = torch.empty(
            (t_samples, 2 * M, M),
            dtype=self.torch_complex_datatype,
            device=torch.device(device),
        )
        d = torch.empty(
            (
                t_samples,
                NT,
            ),
            dtype=self.torch_complex_datatype,
            device=torch.device(device),
        )

        es = []
        qs = []

        # initialize Q-D table
        q = fp[:, 1 : 2 * M + 1] / fp[:, 0 : 2 * M]
        q[:, 0] = fp[:, 1] / (fp[:, 0] / 2.0)
        qs.append(q)

        # rhombus rule for filling triangular Q-D table (e & q)
        for r in range(1, M + 1):
            # start with e, column 1, 0:2*M-2
            mr = 2 * (M - r) + 1
            if r == 1:
                e = torch.zeros(
                    (
                        t_samples,
                        NT,
                    ),
                    dtype=self.torch_complex_datatype,
                    device=torch.device(device),
                )
                e[:, 0:mr] = qs[r - 1][:, 1 : mr + 1] - qs[r - 1][:, 0:mr]
                es.append(e)
            else:
                e = torch.zeros(
                    (
                        t_samples,
                        NT,
                    ),
                    dtype=self.torch_complex_datatype,
                    device=torch.device(device),
                )
                e[:, 0:mr] = (
                    qs[r - 1][:, 1 : mr + 1]
                    - qs[r - 1][:, 0:mr]
                    + es[r - 2][:, 1 : mr + 1]
                )
                es.append(e)
            if not r == M:
                q = torch.zeros(
                    (
                        t_samples,
                        2 * M,
                    ),
                    dtype=self.torch_complex_datatype,
                    device=torch.device(device),
                )
                q[:, 0:mr] = (
                    qs[r - 1][:, 1 : mr + 1]
                    * es[r - 1][:, 1 : mr + 1]
                    / es[r - 1][:, 0:mr]
                )
                qs.append(q)

        # build up continued fraction coefficients (d)
        d[:, 0] = fp[:, 0] / 2.0
        for r in range(1, M + 1):
            d[:, 2 * r - 1] = -qs[r - 1][:, 0]  # even terms
            d[:, 2 * r] = -es[r - 1][:, 0]  # odd terms

        # seed A and B for recurrence
        Aim1 = torch.view_as_complex(torch.Tensor([0, 0]).to(device))
        Ai = d[:, 0]
        Bim1 = torch.view_as_complex(torch.Tensor([1, 0]).to(device))
        Bi = torch.view_as_complex(torch.Tensor([1, 0]).to(device))

        # base of the power series
        z = torch.exp(1j * torch.pi * t / T)

        # coefficients of Pade approximation (A & B)
        # using recurrence for all but last term
        for i in range(1, 2 * M):
            Ait = Ai + d[:, i] * Aim1 * z
            Aim1 = Ai
            Ai = Ait
            Bit = Bi + d[:, i] * Bim1 * z
            Bim1 = Bi
            Bi = Bit

        # "improved remainder" to continued fraction
        brem = (1.0 + (d[:, 2 * M - 1] - d[:, 2 * M]) * z) / 2.0
        rem = -brem * (1.0 - torch.sqrt(1.0 + d[:, 2 * M] * z / brem**2))

        # last term of recurrence using new remainder
        A_2mp1 = Ai + rem * Aim1
        B_2mp1 = Bi + rem * Bim1

        # diagonal Pade approximation
        # F=A/B represents accelerated trapezoid rule
        result = (torch.exp(gamma * t) / T) * (A_2mp1 / B_2mp1).real
        return result.real


class Fourier(InverseLaplaceTransformAlgorithmBase):
    r"""Inherits from :meth:`torchlaplace.inverse_laplace.InverseLaplaceTransformAlgorithmBase`.

    Reconstruct trajectories :math:`\mathbf{x}(t)` for a system of Laplace representations.
    Given a parameterized Laplace representation function :math:`\mathbf{F}(\mathbf{s})`.

    Expands ILT Equation into an expanded Fourier transform, approximating it with the trapezoidal rule.
    This keeps the Bromwich contour parallel to the imaginary axis, and shifts it along the real axis, i.e. :math:`\sigma \propto \frac{1}{t}`.
    It is fairly easy to implement and scale to multiple dimensions. We denote :math:`s=\sigma + i \omega` and we can express it as,

    .. math::
        \begin{split}
            \mathbf{x}(t) & =  \frac{1}{\pi}e^{\sigma t} \int_0^\infty \Re \left\{ F(s) e^{i \omega t} \right\} d \omega \\
            & \approx \frac{1}{T} e^{\sigma t} \left[ \frac{F(\sigma)}{2}  + \sum_{k=1}^{2N} \Re \left\{  F \left( \sigma + \frac{ik\pi}{T} \right)e^{\frac{ik \pi t}{T}} \right\}      \right]
        \end{split}

    Where we approximate the first Fourier ILT, Equation as a discretized version, using the trapezoidal rule with step size :math:`\frac{\pi}{T}` and evaluating :math:`s` at the
    approximation points :math:`s_k=\sigma + \frac{ik\pi}{T}` in the trapezoidal summation.
    We set the parameters of :math:`\sigma=\alpha-\frac{\log(\epsilon)}{T}`, with :math:`\alpha=1e-3`, :math:`\epsilon=10\alpha`, and the scaling parameter :math:`T=2t`. This gives the query function,

    .. math::
        \begin{split}
            s_k(t) & = \text{1e-3} - \frac{\log(\text{1e-2})}{2t} + \frac{ik\pi}{2t} \\
            \mathcal{Q}(t) & = [s_0(t), \ldots, s_{2N}(t)]^T
        \end{split}

    Where we model the equation with :math:`2N + 1` reconstruction terms, setting :math:`N=16` in experiments, and use double point floating precision to increase the numerical precision of the ILT.

    The ILT-FSI equation provides guarantees that we can always find the inverse from time :math:`t: 0 \rightarrow \infty`, given that the singularities of the system (i.e. the
    points at which :math:`F(s) \to \infty`) lie left of the contour of integration, and this puts no constraint on the imaginary frequency components we can model.
    Of course in practice, we often do not model time at :math:`\infty` and instead model up to a fixed time in the future, which then bounds the exponentially increasing system
    trajectories, and their associated system poles that we can model :math:`\sigma \propto \frac{1}{t}`.

    Args:
        ilt_reconstruction_terms (int): number of ILT reconstruction terms, i.e. the number of complex :math:`s` points in `fs` to reconstruct a single time point.
        torch_float_datatype (Torch.dtype): Torch float datatype to use internally in the ILT algorithm, and also output data type of :math:`\mathbf{x}(t)`. Default `torch.float32`.
        torch_complex_datatype (Torch.dtype): Torch complex datatype to use internally in the ILT algorithm. Default `torch.cfloat`.
        alpha (float): :math:`\alpha`, default :math:`\alpha=1e-3`.
        tol (float): desired tolerance, if not specified simply related to alpha as :math:`\text{tol}=10\alpha`.
        scale (float): scaling factor (tuneable), default 2.0.
        eps (float): Small machine floating point precision, default :math:`\text{eps}=1e-6`.
    """

    def __init__(
        self,
        ilt_reconstruction_terms=33,
        alpha=1.0e-3,
        tol=None,
        scale=2.0,
        eps=1e-6,
        torch_float_datatype=TORCH_FLOAT_DATATYPE,
        torch_complex_datatype=TORCH_COMPLEX_DATATYPE,
    ):
        # Unaccelerated Fourier Estimate of DeHoog (Ref: F. R. DE HOOG, J. H. KNIGHT AND A. N. STOKES, p. 360)
        super().__init__(
            ilt_reconstruction_terms, torch_float_datatype, torch_complex_datatype
        )
        M = int((ilt_reconstruction_terms - 1) / 2)
        self.M = M
        self.alpha = torch.Tensor([alpha]).to(device)
        # desired tolerance (here simply related to alpha)
        if tol is not None:
            self.tol = torch.Tensor([tol]).to(device)
        else:
            self.tol = self.alpha * 10.0
        self.nt = ilt_reconstruction_terms  # number of terms in approximation
        # scaling factor (likely tune-able, but 2 is typical)
        self.scale = torch.Tensor([scale]).to(device)
        self.k = torch.arange(
            self.nt, dtype=torch_float_datatype, device=torch.device(device)
        )
        self.eps = eps
        self.torch_float_datatype = torch_float_datatype

    def compute_s(self, ti, time_max=None, Ti=None):
        r"""Computes :math:`\mathbf{s}` to evaluate the Laplace representation :math:`\mathbf{F}(\mathbf{s})` at, from the input time points :math:`t`, using the selected ILT algorithm.

        Args:
            ti (Tensor): time points to reconstruct trajectory for of shape :math:`(\text{SeqLen})`.
            time_max (float): Maximum time to compute reconstruction up to. Best results use default of None, as this uses `time_max=ti`.
            Ti (float): Scaled maximum time to compute reconstruction up to. Best results use default of None, as this uses `T=time_max * self.scale`.

        Returns:
            Tensor of complex s points :math:`\mathbf{s}` of shape :math:`(\text{SeqLen}, \text{ReconTerms})`. :math:`\text{SeqLen}` dimension corresponds to the different time points :math:`t`.

        """
        t = real_vector_to_complex(ti)
        if time_max is not None:
            tmax = torch.Tensor([time_max]).to(device) * torch.ones_like(t)
        else:
            tmax = t  # + self.eps
        if Ti is not None:
            if type(Ti) != self.torch_float_datatype:
                raise ValueError("Invalid Ti shapes")
            T = Ti
        else:
            T = (tmax * self.scale).to(self.torch_complex_datatype)
        gamma = self.alpha - torch.log(self.tol) / (self.scale * T)
        si = torch.matmul(1 / T.view(-1, 1), 1j * torch.pi * self.k.view(1, -1))
        return gamma.view(-1, 1) + si, T

    def line_integrate(self, fp, ti, T):
        r"""Reconstruct trajectories :math:`\mathbf{x}(t)` for `fp`, Laplace representations evaluated at `s` points from the input `ti` points, :math:`t`, using the selected ILT algorithm.

        Args:
            fp (Tensor): Laplace representation evaluated at `s` points derived from the input time points `ti`. `fp` has shape :math:`(\text{SeqLen}, d_{\text{obs}}, \text{ReconTerms})`
            ti (Tensor): time points to reconstruct trajectory for of shape :math:`(\text{SeqLen})`.
            T (Tensor): time points to reconstruct trajectory for used as the reconstruction of times up to `T` time point of shape :math:`(\text{SeqLen})`. Best practice for
            most ILT algorithms is to set `T = ti`, for the ILT algorithms that rely on `T`.

        Returns:
            Tensor of reconstructions :math:`\mathbf{x}(t)` of shape :math:`(\text{SeqLen}, d_{\text{obs}})`. :math:`\text{SeqLen}` dimension corresponds to the different time
            points :math:`t`. This tensor of reconstructions contains the solved value of :math:`\mathbf{x}` for each desired time point in `t`.

        """
        t = real_vector_to_complex(ti)
        gamma = self.alpha - torch.log(self.tol) / (self.scale * T)
        return (
            (1 / T)
            * torch.exp(gamma * t)
            * (
                fp[:, 0] / 2.0
                + torch.sum(
                    fp[:, 1:]
                    * torch.exp(
                        torch.matmul(
                            (t / T).view(-1, 1), 1j * self.k[1:].view(1, -1) * torch.pi
                        )
                    ),
                    1,
                )
            )
        ).real

    def line_integrate_all_multi(self, fp, ti, T):
        r"""Reconstruct trajectories :math:`\mathbf{x}(t)` for `fp`, Laplace representations evaluated at `s` points from the input `ti` points, :math:`t`, using the selected ILT algorithm (takes batch input of `fp`).

        Args:
            fp (Tensor): Laplace representation evaluated at `s` points derived from the input time points `ti`. `fp` has shape :math:`(\text{BatchSize}, \text{SeqLen}, d_{\text{obs}}, \text{ReconTerms})`
            ti (Tensor): time points to reconstruct trajectory for of shape :math:`(\text{SeqLen})`.
            T (Tensor): time points to reconstruct trajectory for used as the reconstruction of times up to `T` time point of shape :math:`(\text{SeqLen})`. Best practice for most ILT algorithms is
            to set `T = ti`, for the ILT algorithms that rely on `T`.

        Returns:
            Tensor of reconstructions :math:`\mathbf{x}(t)` of shape :math:`(\text{BatchSize}, \text{SeqLen}, d_{\text{obs}})`. :math:`\text{SeqLen}` dimension corresponds to the different time
            points :math:`t`. This tensor of reconstructions contains the solved value of :math:`\mathbf{x}` for each desired time point in `t`.

        """
        # Input shapes
        # fp [batch, times, data_dims, approximation_terms * 2 + 1]
        # ti [times]
        # T [times]
        # Returns
        # [batch, times, data_dims]
        t = real_vector_to_complex(ti)  # Could be slowing down ?
        gamma = self.alpha - torch.log(self.tol) / (self.scale * T)
        return (
            ((1 / T) * torch.exp(gamma * t)).view(1, -1, 1)
            * (
                fp[:, :, :, 0] / 2.0
                + torch.sum(
                    fp[:, :, :, 1:]
                    * torch.exp(
                        torch.matmul(
                            (t / T).view(-1, 1), 1j * self.k[1:].view(1, -1) * torch.pi
                        )
                    ).view(fp.shape[1], 1, fp.shape[3] - 1),
                    3,
                )
            )
        ).real

    def line_integrate_all_multi_batch_time(self, fp, ti, T):
        r"""Reconstruct trajectories :math:`\mathbf{x}(t)` for `fp`, Laplace representations evaluated at `s` points from the input `ti` points, :math:`t`, using the selected ILT algorithm (takes batch input of `fp`).

        Args:
            fp (Tensor): Laplace representation evaluated at `s` points derived from the input time points `ti`. `fp` has shape :math:`(\text{BatchSize}, \text{SeqLen}, d_{\text{obs}}, \text{ReconTerms})`
            ti (Tensor): time points to reconstruct trajectory for of shape :math:`(\text{BatchSize}, \text{SeqLen})`.
            T (Tensor): time points to reconstruct trajectory for used as the reconstruction of times up to `T` time point of shape :math:`(\text{SeqLen})`. Best practice for most ILT algorithms is
            to set `T = ti`, for the ILT algorithms that rely on `T`.

        Returns:
            Tensor of reconstructions :math:`\mathbf{x}(t)` of shape :math:`(\text{BatchSize}, \text{SeqLen}, d_{\text{obs}})`. :math:`\text{SeqLen}` dimension corresponds to the different time
            points :math:`t`. This tensor of reconstructions contains the solved value of :math:`\mathbf{x}` for each desired time point in `t`.

        """
        # Input shapes
        # fp [batch, times, data_dims, approximation_terms * 2 + 1]
        # ti [times]
        # T [times]
        # Returns
        # [batch, times, data_dims]
        t = real_vector_to_complex(ti)  # Could be slowing down ?
        gamma = self.alpha - torch.log(self.tol) / (self.scale * T)
        return (
            ((1 / T) * torch.exp(gamma * t)).view(fp.shape[0], fp.shape[1], 1)
            * (
                fp[:, :, :, 0] / 2.0
                + torch.sum(
                    fp[:, :, :, 1:]
                    * torch.exp(
                        torch.matmul(
                            (t / T).view(-1, 1), 1j * self.k[1:].view(1, -1) * torch.pi
                        )
                    ).view(fp.shape[0], fp.shape[1], 1, fp.shape[3] - 1),
                    3,
                )
            )
        ).real

    def line_integrate_multi(self, fp, ti, T):
        r"""Reconstruct trajectories :math:`\mathbf{x}(t)` for `fp`, Laplace representations evaluated at `s` points from the input `ti` points, :math:`t`, using the selected ILT algorithm.

        Args:
            fp (Tensor): Laplace representation evaluated at `s` points derived from the input time points `ti`. `fp` has shape :math:`(\text{SeqLen}, d_{\text{obs}}, \text{ReconTerms})`
            ti (Tensor): time points to reconstruct trajectory for of shape :math:`(\text{SeqLen})`.
            T (Tensor): time points to reconstruct trajectory for used as the reconstruction of times up to `T` time point of shape :math:`(\text{SeqLen})`. Best practice for
            most ILT algorithms is to set `T = ti`, for the ILT algorithms that rely on `T`.

        Returns:
            Tensor of reconstructions :math:`\mathbf{x}(t)` of shape :math:`(\text{SeqLen}, d_{\text{obs}})`. :math:`\text{SeqLen}` dimension corresponds to the different time
            points :math:`t`. This tensor of reconstructions contains the solved value of :math:`\mathbf{x}` for each desired time point in `t`.

        """
        t = real_vector_to_complex(ti)
        gamma = self.alpha - torch.log(self.tol) / (self.scale * T)
        return (
            (1 / T)
            * torch.exp(gamma * t)
            * (
                fp[:, :, 0] / 2.0
                + torch.sum(
                    fp[:, :, 1:]
                    * torch.exp(
                        torch.matmul(
                            (t / T).view(-1, 1), 1j * self.k[1:].view(1, -1) * torch.pi
                        )
                    ),
                    2,
                )
            )
        ).real

    def forward(self, fs, ti, time_max=None, Ti=None):
        r"""Reconstructs a trajectory :math:`\mathbf{x}(t)` for a Laplace representation :math:`\mathbf{F}(\mathbf{s})`, at time points :math:`t`.

        Args:
            fs (Torch.nn.Module or Callable): The first parameter.
            ti (Tensor): time points to reconstruct trajectory for of shape :math:`(\text{SeqLen})`.
            time_max (float): Maximum time to compute reconstruction up to. Best results use default of None, as this uses `time_max=ti`.
            Ti (float): Scaled maximum time to compute reconstruction up to. Best results use default of None, as this uses `T=time_max * self.scale`.

        Returns:
            Tensor of reconstructions :math:`\mathbf{x}(t)` of shape :math:`(\text{SeqLen}, d_{\text{obs}})`. :math:`\text{SeqLen}` dimension corresponds to the different
            time points :math:`t`. This tensor of reconstructions contains the solved value of :math:`\mathbf{x}` for each desired time point in `t`.

        """
        # fs: C^1 -> C^1
        t = real_vector_to_complex(ti)
        if time_max is not None:
            tmax = torch.Tensor([time_max]).to(device) * torch.ones_like(t)
        else:
            tmax = t
        if Ti is not None:
            if type(Ti) != self.torch_float_datatype:
                raise ValueError("Invalid Ti type")
            T = Ti
        else:
            T = (tmax * self.scale).to(self.torch_complex_datatype)
        gamma = self.alpha - torch.log(self.tol) / (self.scale * T)
        si = torch.matmul(1 / T.view(-1, 1), 1j * torch.pi * self.k.view(1, -1))
        fp = fs(gamma.view(-1, 1) + si)
        return (
            (1 / T)
            * torch.exp(gamma * t)
            * (
                fp[:, 0] / 2.0
                + torch.sum(
                    fp[:, 1:]
                    * torch.exp(
                        torch.matmul(
                            (t / T).view(-1, 1), 1j * self.k[1:].view(1, -1) * torch.pi
                        )
                    ),
                    1,
                )
            )
        ).real


class Gaver(InverseLaplaceTransformAlgorithmBase):
    r"""Inherits from :meth:`torchlaplace.inverse_laplace.InverseLaplaceTransformAlgorithmBase`.

    Reconstruct trajectories :math:`\mathbf{x}(t)` for a system of Laplace representations.
    Given a parameterized Laplace representation function :math:`\mathbf{F}(\mathbf{s})`.

    Gaver method, uses a similar form to that of the Fourier Series Inverse, approximating ILT Equation with the trapezoidal rule. This uses the form of,

    .. math::
        \begin{aligned}
            \mathbf{x}(t) \approx \frac{1}{T}\sum_{k=1}^{2N} \eta_k F \left(\frac{\beta_k}{T}\right)
        \end{aligned}

    The coefficients :math:`\eta_k, \beta_k` are determined by the Gaver procedure, in the Abate-Whitt framework.

    .. note::
        Reference: Horváth, G., Horváth, I., Almousa, S. A.-D., and Telek, M. Numerical inverse laplace transformation using concentrated matrix exponential distributions. Performance Evaluation, 137:102067, 2020.

    Args:
        ilt_reconstruction_terms (int): number of ILT reconstruction terms, i.e. the number of complex :math:`s` points in `fs` to reconstruct a single time point.
        torch_float_datatype (Torch.dtype): Torch float datatype to use internally in the ILT algorithm, and also output data type of :math:`\mathbf{x}(t)`. Default `torch.float32`.
        torch_complex_datatype (Torch.dtype): Torch complex datatype to use internally in the ILT algorithm. Default `torch.cfloat`.
    """

    def __init__(
        self,
        ilt_reconstruction_terms=33,
        torch_float_datatype=TORCH_FLOAT_DATATYPE,
        torch_complex_datatype=TORCH_COMPLEX_DATATYPE,
    ):
        super().__init__(
            ilt_reconstruction_terms, torch_float_datatype, torch_complex_datatype
        )
        M = int((ilt_reconstruction_terms - 1) / 2)
        self.M = M
        max_fn_evals = ilt_reconstruction_terms
        if max_fn_evals % 2 == 1:
            max_fn_evals -= 1
        ndiv2 = int(max_fn_evals / 2)
        eta = np.zeros(max_fn_evals)
        beta = np.zeros(max_fn_evals)
        logsum = np.concatenate(
            ([0], np.cumsum(np.log(np.arange(1, max_fn_evals + 1))))
        )
        for k in range(1, max_fn_evals + 1):
            inside_sum = 0.0
            for j in range(np.floor((k + 1) / 2).astype(np.int32), min(k, ndiv2) + 1):
                inside_sum += np.exp(
                    (ndiv2 + 1) * np.log(j)
                    - logsum[ndiv2 - j]
                    + logsum[2 * j]
                    - 2 * logsum[j]
                    - logsum[k - j]
                    - logsum[2 * j - k]
                )
            eta[k - 1] = np.log(2.0) * (-1) ** (k + ndiv2) * inside_sum
            beta[k - 1] = k * np.log(2.0)
        self.eta = (
            complex_numpy_to_complex_torch(eta).to(torch_complex_datatype).to(device)
        )
        self.beta = (
            complex_numpy_to_complex_torch(beta).to(torch_complex_datatype).to(device)
        )

    def compute_s(self, ti):
        t = real_vector_to_complex(ti)
        s = torch.matmul((1 / t).view(-1, 1), self.beta.view(1, -1))
        return s, t

    def line_integrate(self, fp, ti):
        # t = real_vector_to_complex(ti)
        return (torch.matmul(fp, self.eta.view(-1, 1)).view(-1) / ti).real

    def line_integrate_all_multi(self, fp, ti, T):
        # Input shapes
        # fp [batch, times, data_dims, approximation_terms * 2 + 1]
        # ti [times]
        # T [times]
        # Returns
        # [batch, times, data_dims]
        # t = real_vector_to_complex(ti)
        return (
            (1 / T).view(1, -1, 1) * torch.matmul(fp, self.eta.view(-1, 1)).squeeze(-1)
        ).real

    def line_integrate_all_multi_batch_time(self, fp, ti, T):
        r"""Reconstruct trajectories :math:`\mathbf{x}(t)` for `fp`, Laplace representations evaluated at `s` points from the input `ti` points, :math:`t`, using the selected ILT algorithm (takes batch input of `fp`).

        Args:
            fp (Tensor): Laplace representation evaluated at `s` points derived from the input time points `ti`. `fp` has shape :math:`(\text{BatchSize}, \text{SeqLen}, d_{\text{obs}}, \text{ReconTerms})`
            ti (Tensor): time points to reconstruct trajectory for of shape :math:`(\text{BatchSize}, \text{SeqLen})`.
            T (Tensor): time points to reconstruct trajectory for used as the reconstruction of times up to `T` time point of shape :math:`(\text{SeqLen})`. Best practice for most ILT algorithms is
            to set `T = ti`, for the ILT algorithms that rely on `T`.

        Returns:
            Tensor of reconstructions :math:`\mathbf{x}(t)` of shape :math:`(\text{BatchSize}, \text{SeqLen}, d_{\text{obs}})`. :math:`\text{SeqLen}` dimension corresponds to the different time
            points :math:`t`. This tensor of reconstructions contains the solved value of :math:`\mathbf{x}` for each desired time point in `t`.

        """
        # Input shapes
        # fp [batch, times, data_dims, approximation_terms * 2 + 1]
        # ti [times]
        # T [times]
        # Returns
        # [batch, times, data_dims]
        return (
            (1 / T).view(T.shape[0], T.shape[1], 1)
            * torch.matmul(fp, self.eta.view(-1, 1)).view(
                fp.shape[0], fp.shape[1], fp.shape[2]
            )
        ).real

    def forward(self, fs, ti):
        # fs: C^1 -> C^1
        t = real_vector_to_complex(ti)
        s = torch.matmul((1 / t).view(-1, 1), self.beta.view(1, -1))
        return (torch.matmul(fs(s), self.eta.view(-1, 1)).view(-1) / t).real


class Euler(InverseLaplaceTransformAlgorithmBase):
    r"""Inherits from :meth:`torchlaplace.inverse_laplace.InverseLaplaceTransformAlgorithmBase`.

    Reconstruct trajectories :math:`\mathbf{x}(t)` for a system of Laplace representations.
    Given a parameterized Laplace representation function :math:`\mathbf{F}(\mathbf{s})`.

    Euler method, uses a similar form to that of the Fourier Series Inverse, approximating ILT Equation with the trapezoidal rule. This uses the form of,

    .. math::
        \begin{aligned}
            \mathbf{x}(t) \approx \frac{1}{T}\sum_{k=1}^{2N} \eta_k F \left(\frac{\beta_k}{T}\right)
        \end{aligned}

    The coefficients :math:`\eta_k, \beta_k` are determined by the Euler procedure, in the Abate-Whitt framework.

    .. note::
        Reference: Horváth, G., Horváth, I., Almousa, S. A.-D., and Telek, M. Numerical inverse laplace transformation using concentrated matrix exponential distributions. Performance Evaluation, 137:102067, 2020.

    Args:
        ilt_reconstruction_terms (int): number of ILT reconstruction terms, i.e. the number of complex :math:`s` points in `fs` to reconstruct a single time point.
        torch_float_datatype (Torch.dtype): Torch float datatype to use internally in the ILT algorithm, and also output data type of :math:`\mathbf{x}(t)`. Default `torch.float32`.
        torch_complex_datatype (Torch.dtype): Torch complex datatype to use internally in the ILT algorithm. Default `torch.cfloat`.
    """

    def __init__(
        self,
        ilt_reconstruction_terms=33,
        torch_float_datatype=TORCH_FLOAT_DATATYPE,
        torch_complex_datatype=TORCH_COMPLEX_DATATYPE,
    ):
        super().__init__(
            ilt_reconstruction_terms, torch_float_datatype, torch_complex_datatype
        )
        M = int((ilt_reconstruction_terms - 1) / 2)
        self.M = M
        max_fn_evals = ilt_reconstruction_terms
        n_euler = np.floor((max_fn_evals - 1) / 2).astype(np.int32)
        end_element = torch.tensor(
            [2.0], dtype=self.torch_float_datatype, device=torch.device(device)
        )
        end_element = end_element.pow(-n_euler)
        eta = torch.concat(
            (
                torch.tensor(
                    [0.5], dtype=self.torch_float_datatype, device=torch.device(device)
                ),
                torch.ones(
                    n_euler,
                    dtype=self.torch_complex_datatype,
                    device=torch.device(device),
                ),
                torch.zeros(
                    n_euler - 1,
                    dtype=self.torch_complex_datatype,
                    device=torch.device(device),
                ),
                end_element,
            )
        )
        logsum = torch.cumsum(
            torch.log(
                torch.arange(
                    1,
                    n_euler + 1,
                    dtype=torch_float_datatype,
                    device=torch.device(device),
                )
            ),
            dim=0,
        )
        for k in torch.arange(1, n_euler, device=torch.device(device)):
            eta[2 * n_euler - k] = eta[2 * n_euler - k + 1] + torch.exp(
                logsum[n_euler - 1]
                - n_euler
                * torch.log(
                    torch.tensor(
                        2.0,
                        dtype=self.torch_float_datatype,
                        device=torch.device(device),
                    )
                )
                - logsum[k - 1]
                - logsum[n_euler - k - 1]
            )
        k = torch.arange(2 * n_euler + 1, device=torch.device(device))
        beta = (
            n_euler
            * torch.log(
                torch.tensor(
                    10.0, dtype=self.torch_float_datatype, device=torch.device(device)
                )
            )
            / 3.0
            + 1j * torch.pi * k
        )
        final_element = torch.tensor(
            10.0, dtype=self.torch_float_datatype, device=torch.device(device)
        )
        final_element = final_element.pow((n_euler) / 3.0)
        eta = final_element * (1 - (k % 2) * 2) * eta
        self.eta = eta.detach().to(torch_complex_datatype)
        self.beta = beta.detach().to(torch_complex_datatype)
        # eta = np.concatenate(([0.5], np.ones(n_euler), np.zeros(n_euler-1), [2.0**-n_euler]))
        # logsum = np.cumsum(np.log(np.arange(1,n_euler+1)))
        # for k in range(1,n_euler):
        #     eta[2*n_euler-k] = eta[2*n_euler-k + 1] + np.exp(logsum[n_euler-1] - n_euler*np.log(2.0) - logsum[k-1] - logsum[n_euler-k-1])
        # k = np.arange(2*n_euler+1)
        # beta = n_euler*np.log(10.0)/3.0 + 1j*np.pi*k
        # eta  = (10**((n_euler)/3.0))*(1-(k%2)*2) * eta
        # self.eta = (
        #     complex_numpy_to_complex_torch(eta).to(torch_complex_datatype).to(device)
        # )
        # self.beta = (
        #     complex_numpy_to_complex_torch(beta).to(torch_complex_datatype).to(device)
        # )

    def compute_s(self, ti):
        t = real_vector_to_complex(ti)
        s = torch.matmul((1 / t).view(-1, 1), self.beta.view(1, -1))
        return s, t

    def line_integrate(self, fp, ti):
        # t = real_vector_to_complex(ti)
        return (torch.matmul(fp, self.eta.view(-1, 1)).view(-1) / ti).real

    def line_integrate_all_multi(self, fp, ti, T):
        # Input shapes
        # fp [batch, times, data_dims, approximation_terms * 2 + 1]
        # ti [times]
        # T [times]
        # Returns
        # [batch, times, data_dims]
        # t = real_vector_to_complex(ti)
        return (
            (1 / T).view(1, -1, 1) * torch.matmul(fp, self.eta.view(-1, 1)).squeeze(-1)
        ).real

    def line_integrate_all_multi_batch_time(self, fp, ti, T):
        r"""Reconstruct trajectories :math:`\mathbf{x}(t)` for `fp`, Laplace representations evaluated at `s` points from the input `ti` points, :math:`t`, using the selected ILT algorithm (takes batch input of `fp`).

        Args:
            fp (Tensor): Laplace representation evaluated at `s` points derived from the input time points `ti`. `fp` has shape :math:`(\text{BatchSize}, \text{SeqLen}, d_{\text{obs}}, \text{ReconTerms})`
            ti (Tensor): time points to reconstruct trajectory for of shape :math:`(\text{BatchSize}, \text{SeqLen})`.
            T (Tensor): time points to reconstruct trajectory for used as the reconstruction of times up to `T` time point of shape :math:`(\text{SeqLen})`. Best practice for most ILT algorithms is
            to set `T = ti`, for the ILT algorithms that rely on `T`.

        Returns:
            Tensor of reconstructions :math:`\mathbf{x}(t)` of shape :math:`(\text{BatchSize}, \text{SeqLen}, d_{\text{obs}})`. :math:`\text{SeqLen}` dimension corresponds to the different time
            points :math:`t`. This tensor of reconstructions contains the solved value of :math:`\mathbf{x}` for each desired time point in `t`.

        """
        # Input shapes
        # fp [batch, times, data_dims, approximation_terms * 2 + 1]
        # ti [times]
        # T [times]
        # Returns
        # [batch, times, data_dims]
        return (
            (1 / T).view(T.shape[0], T.shape[1], 1)
            * torch.matmul(fp, self.eta.view(-1, 1)).view(
                fp.shape[0], fp.shape[1], fp.shape[2]
            )
        ).real

    def forward(self, fs, ti):
        # fs: C^1 -> C^1
        t = real_vector_to_complex(ti)
        s = torch.matmul((1 / t).view(-1, 1), self.beta.view(1, -1))
        return (torch.matmul(fs(s), self.eta.view(-1, 1)).view(-1) / t).real


class CME(InverseLaplaceTransformAlgorithmBase):
    r"""Inherits from :meth:`torchlaplace.inverse_laplace.InverseLaplaceTransformAlgorithmBase`.

    Reconstruct trajectories :math:`\mathbf{x}(t)` for a system of Laplace representations.
    Given a parameterized Laplace representation function :math:`\mathbf{F}(\mathbf{s})`.

    Concentrated matrix exponential (CME), uses a similar form to that of the Fourier Series Inverse, approximating ILT Equation with the trapezoidal rule. This uses the form of,

    .. math::
        \begin{aligned}
            \mathbf{x}(t) \approx \frac{1}{T}\sum_{k=1}^{2N} \eta_k F \left(\frac{\beta_k}{T}\right)
        \end{aligned}

    The coefficients :math:`\eta_k, \beta_k` are determined by a complex procedure, with a numerical optimization step involved.
    This provides a good approximation for the reconstruction and the coefficients of up to a pre-specified order can be pre-computed and cached for low complexity run time.
    Similarly to Fourier (FSI), CMEs Bromwich contour remains parallel to the imaginary axis and is shifted along the real axis, i.e. :math:`\sigma \propto \frac{1}{t}`.
    It is moderately easy to implement when using pre-computed coefficients and scale to multiple dimensions.

    .. note::
        Reference: Horváth, G., Horváth, I., Almousa, S. A.-D., and Telek, M. Numerical inverse laplace transformation using concentrated matrix exponential distributions. Performance Evaluation, 137:102067, 2020.

    Args:
        ilt_reconstruction_terms (int): number of ILT reconstruction terms, i.e. the number of complex :math:`s` points in `fs` to reconstruct a single time point.
        torch_float_datatype (Torch.dtype): Torch float datatype to use internally in the ILT algorithm, and also output data type of :math:`\mathbf{x}(t)`. Default `torch.float32`.
        torch_complex_datatype (Torch.dtype): Torch complex datatype to use internally in the ILT algorithm. Default `torch.cfloat`.
    """

    def __init__(
        self,
        ilt_reconstruction_terms=33,
        torch_float_datatype=TORCH_FLOAT_DATATYPE,
        torch_complex_datatype=TORCH_COMPLEX_DATATYPE,
    ):
        super().__init__(
            ilt_reconstruction_terms, torch_float_datatype, torch_complex_datatype
        )
        M = int((ilt_reconstruction_terms - 1) / 2)
        self.M = M
        cme_params = cme_params_factory()
        params = cme_params[0]
        max_fn_evals = ilt_reconstruction_terms
        for p in cme_params:
            if p["cv2"] < params["cv2"] and p["n"] + 1 <= max_fn_evals:
                params = p
        eta = (
            np.concatenate(
                ([params["c"]], np.array(params["a"]) + 1j * np.array(params["b"]))
            )
            * params["mu1"]
        )
        beta = (
            np.concatenate(
                ([1], 1 + 1j * np.arange(1, params["n"] + 1) * params["omega"])
            )
            * params["mu1"]
        )
        self.eta = (
            complex_numpy_to_complex_torch(eta).to(torch_complex_datatype).to(device)
        )
        self.beta = (
            complex_numpy_to_complex_torch(beta).to(torch_complex_datatype).to(device)
        )

    def compute_s(self, ti):
        t = real_vector_to_complex(ti)
        s = torch.matmul((1 / t).view(-1, 1), self.beta.view(1, -1))
        return s, t

    def line_integrate(self, fp, ti):
        # t = real_vector_to_complex(ti)
        return (torch.matmul(fp, self.eta.view(-1, 1)).view(-1) / ti).real

    def line_integrate_all_multi(self, fp, ti, T):
        # Input shapes
        # fp [batch, times, data_dims, approximation_terms * 2 + 1]
        # ti [times]
        # T [times]
        # Returns
        # [batch, times, data_dims]
        # t = real_vector_to_complex(ti)
        return (
            (1 / T).view(1, -1, 1) * torch.matmul(fp, self.eta.view(-1, 1)).squeeze(-1)
        ).real

    def line_integrate_all_multi_batch_time(self, fp, ti, T):
        r"""Reconstruct trajectories :math:`\mathbf{x}(t)` for `fp`, Laplace representations evaluated at `s` points from the input `ti` points, :math:`t`, using the selected ILT algorithm (takes batch input of `fp`).

        Args:
            fp (Tensor): Laplace representation evaluated at `s` points derived from the input time points `ti`. `fp` has shape :math:`(\text{BatchSize}, \text{SeqLen}, d_{\text{obs}}, \text{ReconTerms})`
            ti (Tensor): time points to reconstruct trajectory for of shape :math:`(\text{BatchSize}, \text{SeqLen})`.
            T (Tensor): time points to reconstruct trajectory for used as the reconstruction of times up to `T` time point of shape :math:`(\text{SeqLen})`. Best practice for most ILT algorithms is
            to set `T = ti`, for the ILT algorithms that rely on `T`.

        Returns:
            Tensor of reconstructions :math:`\mathbf{x}(t)` of shape :math:`(\text{BatchSize}, \text{SeqLen}, d_{\text{obs}})`. :math:`\text{SeqLen}` dimension corresponds to the different time
            points :math:`t`. This tensor of reconstructions contains the solved value of :math:`\mathbf{x}` for each desired time point in `t`.

        """
        # Input shapes
        # fp [batch, times, data_dims, approximation_terms * 2 + 1]
        # ti [times]
        # T [times]
        # Returns
        # [batch, times, data_dims]
        return (
            (1 / T).view(T.shape[0], T.shape[1], 1)
            * torch.matmul(fp, self.eta.view(-1, 1)).view(
                fp.shape[0], fp.shape[1], fp.shape[2]
            )
        ).real

    def forward(self, fs, ti):
        # fs: C^1 -> C^1
        t = real_vector_to_complex(ti)
        s = torch.matmul((1 / t).view(-1, 1), self.beta.view(1, -1))
        return (torch.matmul(fs(s), self.eta.view(-1, 1)).view(-1) / t).real


if __name__ == "__main__":
    ILT_RECONSTRUCTION_TERMS = 33

    t = torch.linspace(0.0001, 10.0, 1000).to(device)

    # Exponential
    # sign = +1  # (-1 for increasing exponential, +1 for decaying exponential)
    # def fs(so): return 1 / (so + 1 * sign)  # Laplace solution
    # def ft(t): return torch.exp(-t * sign)  # Time solution

    # # Cosine
    def fs(so):
        return so / (so**2 + 1)  # Laplace solution

    def ft(t):
        return torch.cos(t)  # Time solution

    print("")

    # Tablot

    # Evaluate s points per time input (Default, as more accurate inversion)

    decoder = FixedTablot(ilt_reconstruction_terms=ILT_RECONSTRUCTION_TERMS).to(device)
    t0 = time()
    f_hat_t = decoder(fs, t)
    print(
        "FixedTablot Loss:\t{}\t\t| time: {}".format(
            np.sqrt(torch.nn.MSELoss()(ft(t), f_hat_t).cpu().numpy()), time() - t0
        )
    )

    # Split evaluation of s points out from that of the line integral (should be the exact same result as above)
    decoder = FixedTablot(ilt_reconstruction_terms=ILT_RECONSTRUCTION_TERMS).to(device)
    t0 = time()
    s, _ = decoder.compute_s(t)
    fh = fs(s)
    f_hat_t = decoder.line_integrate(fh, t)
    print(
        "FixedTablot Loss (Split apart):\t{}\t| time: {}".format(
            np.sqrt(torch.nn.MSELoss()(ft(t), f_hat_t).cpu().numpy()), time() - t0
        )
    )

    decoder = FixedTablot(ilt_reconstruction_terms=ILT_RECONSTRUCTION_TERMS).to(device)
    t0 = time()
    s, _ = decoder.compute_s(t, time_max=torch.max(t))
    fh = fs(s)
    f_hat_t = decoder.line_integrate(fh, t, time_max=t.max().item())
    print(
        "FixedTablot Loss (Split apart, Fixed Max Time):\t{}\t| time: {}".format(
            np.sqrt(torch.nn.MSELoss()(ft(t), f_hat_t).cpu().numpy()), time() - t0
        )
    )

    # Evaluate s points for one fixed time, maximum time (Less accurate, maybe more stable ?)

    decoder = FixedTablot(ilt_reconstruction_terms=ILT_RECONSTRUCTION_TERMS).to(device)
    t0 = time()
    f_hat_t = decoder(fs, t, time_max=torch.max(t))
    print(
        "FixedTablot Loss (Fixed Max Time):\t{}\t| time: {}".format(
            np.sqrt(torch.nn.MSELoss()(ft(t), f_hat_t).cpu().numpy()), time() - t0
        )
    )

    # Stehfest - Increasing degree here, introduces numerical error that increases larger than other methods, therefore for high degree becomes unstable

    decoder = Stehfest(ilt_reconstruction_terms=ILT_RECONSTRUCTION_TERMS).to(device)
    t0 = time()
    f_hat_t = decoder(fs, t)
    print(
        "Stehfest Loss:\t{}\t| time: {}".format(
            np.sqrt(torch.nn.MSELoss()(ft(t), f_hat_t).cpu().numpy()), time() - t0
        )
    )

    decoder = Stehfest(ilt_reconstruction_terms=ILT_RECONSTRUCTION_TERMS).to(device)
    t0 = time()
    s = decoder.compute_s(t)
    fh = fs(s)
    f_hat_t = decoder.line_integrate(fh, t)
    print(
        "Stehfest Loss (Split apart):\t{}\t| time: {}".format(
            np.sqrt(torch.nn.MSELoss()(ft(t), f_hat_t).cpu().numpy()), time() - t0
        )
    )

    # Fourier (Un accelerated DeHoog)
    decoder = Fourier(ilt_reconstruction_terms=ILT_RECONSTRUCTION_TERMS).to(device)
    t0 = time()
    f_hat_t = decoder(fs, t)
    print(
        "Fourier (Un accelerated DeHoog) Loss:\t{}\t| time: {}".format(
            np.sqrt(torch.nn.MSELoss()(ft(t), f_hat_t).cpu().numpy()), time() - t0
        )
    )

    decoder = Fourier(ilt_reconstruction_terms=ILT_RECONSTRUCTION_TERMS).to(device)
    t0 = time()
    s, T = decoder.compute_s(t)
    fh = fs(s)
    f_hat_t = decoder.line_integrate(fh, t, T)
    print(
        "Fourier (Un accelerated DeHoog) Loss (Split apart):\t{}\t| time: {}".format(
            np.sqrt(torch.nn.MSELoss()(ft(t), f_hat_t).cpu().numpy()), time() - t0
        )
    )

    # DeHoog

    decoder = DeHoog(ilt_reconstruction_terms=ILT_RECONSTRUCTION_TERMS).to(device)
    t0 = time()
    f_hat_t = decoder(fs, t)
    print(
        "DeHoog Loss:\t{}\t| time: {}".format(
            np.sqrt(torch.nn.MSELoss()(ft(t), f_hat_t).cpu().numpy()), time() - t0
        )
    )

    # Split evaluation of s points out from that of the line integral (should be the exact same result as above)
    decoder = DeHoog(ilt_reconstruction_terms=ILT_RECONSTRUCTION_TERMS).to(device)
    t0 = time()
    s, T = decoder.compute_s(t)
    fh = fs(s)
    f_hat_t = decoder.line_integrate(fh, t, T)
    print(
        "DeHoog Loss (Split apart):\t{}\t| time: {}".format(
            np.sqrt(torch.nn.MSELoss()(ft(t), f_hat_t).cpu().numpy()), time() - t0
        )
    )

    # Single line integral
    decoder = DeHoog(ilt_reconstruction_terms=ILT_RECONSTRUCTION_TERMS).to(device)
    t0 = time()
    s = decoder.compute_fixed_s(torch.max(t))
    fh = fs(s)
    f_hat_t = decoder.fixed_line_integrate(fh, t, torch.max(t))
    print(
        "DeHoog Loss (Fixed Line Integrate):\t{}\t| time: {}".format(
            np.sqrt(torch.nn.MSELoss()(ft(t), f_hat_t).cpu().numpy()), time() - t0
        )
    )

    decoder = DeHoog(ilt_reconstruction_terms=ILT_RECONSTRUCTION_TERMS).to(device)
    t0 = time()
    f_hat_t = decoder(fs, t, time_max=torch.max(t))
    print(
        "DeHoog Loss (Fixed Max Time):\t{}\t| time: {}".format(
            np.sqrt(torch.nn.MSELoss()(ft(t), f_hat_t).cpu().numpy()), time() - t0
        )
    )

    # CME
    decoder = CME(ilt_reconstruction_terms=ILT_RECONSTRUCTION_TERMS).to(device)
    t0 = time()
    f_hat_t = decoder(fs, t)
    print(
        "CME Loss:\t{}\t| time: {}".format(
            np.sqrt(torch.nn.MSELoss()(ft(t), f_hat_t).cpu().numpy()), time() - t0
        )
    )

    decoder = CME(ilt_reconstruction_terms=ILT_RECONSTRUCTION_TERMS).to(device)
    t0 = time()
    s, T = decoder.compute_s(t)
    fh = fs(s)
    f_hat_t = decoder.line_integrate(fh, t)
    print(
        "CME Loss (Split apart):\t{}\t| time: {}".format(
            np.sqrt(torch.nn.MSELoss()(ft(t), f_hat_t).cpu().numpy()), time() - t0
        )
    )

    # Euler
    decoder = Euler(ilt_reconstruction_terms=ILT_RECONSTRUCTION_TERMS).to(device)
    t0 = time()
    f_hat_t = decoder(fs, t)
    print(
        "Euler Loss:\t{}\t| time: {}".format(
            np.sqrt(torch.nn.MSELoss()(ft(t), f_hat_t).cpu().numpy()), time() - t0
        )
    )

    decoder = Euler(ilt_reconstruction_terms=ILT_RECONSTRUCTION_TERMS).to(device)
    t0 = time()
    s, T = decoder.compute_s(t)
    fh = fs(s)
    f_hat_t = decoder.line_integrate(fh, t)
    print(
        "Euler Loss (Split apart):\t{}\t| time: {}".format(
            np.sqrt(torch.nn.MSELoss()(ft(t), f_hat_t).cpu().numpy()), time() - t0
        )
    )

    # Gaver
    decoder = Gaver(ilt_reconstruction_terms=ILT_RECONSTRUCTION_TERMS).to(device)
    t0 = time()
    f_hat_t = decoder(fs, t)
    print(
        "Gaver Loss:\t{}\t| time: {}".format(
            np.sqrt(torch.nn.MSELoss()(ft(t), f_hat_t).cpu().numpy()), time() - t0
        )
    )

    decoder = Gaver(ilt_reconstruction_terms=ILT_RECONSTRUCTION_TERMS).to(device)
    t0 = time()
    s, T = decoder.compute_s(t)
    fh = fs(s)
    f_hat_t = decoder.line_integrate(fh, t)
    print(
        "Gaver Loss (Split apart):\t{}\t| time: {}".format(
            np.sqrt(torch.nn.MSELoss()(ft(t), f_hat_t).cpu().numpy()), time() - t0
        )
    )
    print("Done")
