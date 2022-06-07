import torch

from torchlaplace.transformations import spherical_riemann_to_complex, complex_to_spherical_riemann, spherical_to_complex, complex_to_spherical

def test_transformations():
    angle_samples = 32
    EPS = 1e-6  # or 7, 8 or higher and numerical error issues
    phi = torch.linspace(
        -torch.pi / 2 + EPS, torch.pi / 2 - EPS, angle_samples
    ).double()
    theta = torch.linspace(0 + EPS, torch.pi - EPS, angle_samples).double()
    s_real, s_imag = spherical_riemann_to_complex(theta, phi)
    theta_r, phi_r = complex_to_spherical_riemann(s_real, s_imag)

    # Inf point
    s_real = torch.Tensor([torch.inf]).double()
    s_imag = torch.Tensor([torch.inf]).double()
    theta_r, phi_r = complex_to_spherical_riemann(s_real, s_imag)

    # # Zero point
    s_real = torch.Tensor([0]).double()
    s_imag = torch.Tensor([0]).double()
    theta_r, phi_r = complex_to_spherical_riemann(s_real, s_imag)

