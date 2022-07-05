###########################
# Neural Laplace: Learning diverse classes of differential equations in the Laplace domain
# Author: Samuel Holt
###########################
import torch

# Internal functions


def spherical_riemann_to_complex(theta, phi):
    r"""Spherical Riemann stereographic projection coordinates to complex number coordinates. I.e. inverse Spherical Riemann stereographic projection map.

    The inverse transform, :math:`v: \mathcal{D} \rightarrow \mathbb{C}`, is given as

    .. math::
        \begin{aligned}
            s = v(\theta, \phi) = \tan \left( \frac{\phi}{2} + \frac{\pi}{4} \right) e^{i \theta}
        \end{aligned}

    Args:
        theta (Tensor): Spherical Riemann stereographic projection coordinates, shape :math:`\theta` component of shape :math:`(\text{dimension})`.
        phi (Tensor): Spherical Riemann stereographic projection coordinates, shape :math:`\phi` component of shape :math:`(\text{dimension})`.

    Returns:
        Tuple Tensor of real and imaginary components of the complex numbers coordinate, of :math:`(\Re(s), \Im(s))`. Where :math:`s \in \mathbb{C}^d`, with :math:`d` is :math:`\text{dimension}`.

    """
    # dim(phi) == dim(theta), takes in spherical co-ordinates returns comlex real & imaginary parts
    if phi.shape != theta.shape:
        raise ValueError("Invalid phi theta shapes")
    r = torch.tan(phi / 2 + torch.pi / 4)
    s_real = r * torch.cos(theta)
    s_imag = r * torch.sin(theta)
    return s_real, s_imag


def complex_to_spherical_riemann(s_real, s_imag):
    r"""Complex coordinates to to Spherical Riemann stereographic projection coordinates.
    I.e. we can translate any complex number :math:`s\in \mathbb{C}` into a coordinate on the Riemann Sphere :math:`(\theta, \phi) \in \mathcal{D} = (-{\pi}, {\pi}) \times (-\frac{\pi}{2}, \frac{\pi}{2})`, i.e.

    .. math::
        \begin{aligned}
            u(s) = \left( \arctan \left( \frac{\Im(s)}{\Re(s)} \right),\arcsin \left( \frac{|s|^2-1}{|s|^2+1} \right) \right)
        \end{aligned}

    For more details see `[1] <https://arxiv.org/abs/2206.04843>`__.

    Args:
        s_real (Tensor): Real component of the complex tensor, of shape :math:`(\text{dimension})`.
        s_imag (Tensor): Imaginary component of the complex tensor, of shape :math:`(\text{dimension})`.

    Returns:
        Tuple Tensor of :math:`(\theta, \phi)` of complex number in spherical Riemann stereographic projection coordinates. Where the shape  of :math:`\theta, \phi` is of shape :math:`(\text{dimension})`.

    """

    # din(s_real) == dim(s_imag), takes in real & complex parts returns spherical co-ordinates
    if s_real.shape != s_imag.shape:
        raise ValueError("Invalid s_real s_imag shapes")
    s_abs_2 = s_imag**2 + s_real**2
    # Handle points at infinity
    phi_r_int = torch.where(
        torch.isinf(s_abs_2), torch.ones_like(s_abs_2), ((s_abs_2 - 1) / (s_abs_2 + 1))
    )
    phi_r = torch.asin(phi_r_int)
    theta_r = torch.atan2(s_imag, s_real)
    return theta_r, phi_r


# Main functions


def spherical_to_complex(theta, phi):
    r"""Spherical Riemann stereographic projection coordinates to complex number coordinates. I.e. inverse Spherical Riemann stereographic projection map.

    The inverse transform, :math:`v: \mathcal{D} \rightarrow \mathbb{C}`, is given as

    .. math::
        \begin{aligned}
            s = v(\theta, \phi) = \tan \left( \frac{\phi}{2} + \frac{\pi}{4} \right) e^{i \theta}
        \end{aligned}

    This uses :meth:`torchlaplace.transformations.spherical_riemann_to_complex`, however provides maintains the shape of the input tensor and output tensor.

    Args:
        theta (Tensor): Spherical Riemann stereographic projection coordinates, shape :math:`\theta` component of shape :math:`(\text{Shape})`.
        phi (Tensor): Spherical Riemann stereographic projection coordinates, shape :math:`\phi` component of shape :math:`(\text{Shape})`.

    Returns:
        Complex Tensor of the complex numbers coordinate, of shape :math:`(\text{Shape})`.

    """
    # Riemann spherical to complex: R^2 -> C^1
    # dim(phi) == dim(theta), takes in spherical co-ordinates returns comlex real & imaginary parts
    s_real, s_imag = spherical_riemann_to_complex(
        torch.flatten(theta), torch.flatten(phi)
    )
    s = torch.view_as_complex(torch.stack((s_real, s_imag), 1))
    return torch.reshape(s, theta.shape)


def complex_to_spherical(s):
    r"""Complex coordinates to to Spherical Riemann stereographic projection coordinates.
    I.e. we can translate any complex number :math:`s\in \mathbb{C}` into a coordinate on the Riemann Sphere :math:`(\theta, \phi) \in \mathcal{D} = (-{\pi}, {\pi}) \times (-\frac{\pi}{2}, \frac{\pi}{2})`, i.e.

    .. math::
        \begin{aligned}
            u(s) = \left( \arctan \left( \frac{\Im(s)}{\Re(s)} \right),\arcsin \left( \frac{|s|^2-1}{|s|^2+1} \right) \right)
        \end{aligned}

    For more details see `[1] <https://arxiv.org/abs/2206.04843>`__.

    This uses :meth:`torchlaplace.transformations.complex_to_spherical_riemann`, however provides maintains the shape of the input tensor and output tensor.

    Args:
        s (Tensor): Complex tensor, of shape :math:`(\text{Shape})`.

    Returns:
        Tuple Tensor of :math:`(\theta, \phi)` of complex number in spherical Riemann stereographic projection coordinates. Where the shape of :math:`\theta, \phi` each is of shape :math:`(\text{Shape})`.

    """
    # Complex to Riemann spherical: C^1 -> R^2
    # dim(phi) == dim(theta), takes in spherical co-ordinates returns comlex real & imaginary parts
    theta, phi = complex_to_spherical_riemann(
        torch.flatten(s.real), torch.flatten(s.imag)
    )
    theta = torch.reshape(theta, s.real.shape)
    phi = torch.reshape(phi, s.imag.shape)
    return theta, phi


if __name__ == "__main__":
    angle_samples = 32
    EPS = 1e-6  # or 7, 8 or higher and numerical error issues
    phi = torch.linspace(
        -torch.pi / 2 + EPS, torch.pi / 2 - EPS, angle_samples
    ).double()
    print("phi ", phi)
    theta = torch.linspace(0 + EPS, torch.pi - EPS, angle_samples).double()
    print("theta ", theta)

    s_real, s_imag = spherical_riemann_to_complex(theta, phi)

    print("s_real ", s_real)
    print("s_imag ", s_imag)

    theta_r, phi_r = complex_to_spherical_riemann(s_real, s_imag)

    print("theta_r ", theta_r)
    print("phi_r ", phi_r)

    print("Seperate tests\n\n")

    # Inf point
    print("Point at infty")
    s_real = torch.Tensor([torch.inf]).double()
    s_imag = torch.Tensor([torch.inf]).double()
    theta_r, phi_r = complex_to_spherical_riemann(s_real, s_imag)

    print("theta_r ", theta_r)
    print("phi_r ", phi_r)

    # # Zero point
    print("Point at zero")
    s_real = torch.Tensor([0]).double()
    s_imag = torch.Tensor([0]).double()
    theta_r, phi_r = complex_to_spherical_riemann(s_real, s_imag)

    print("theta_r ", theta_r)
    print("phi_r ", phi_r)
