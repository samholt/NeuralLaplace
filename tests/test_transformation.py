"""
Test the transformations between spherical Riemannian coordinates and complex"""
import torch

from torchlaplace.transformations import (
    complex_to_spherical_riemann,
    spherical_riemann_to_complex,
)


def test_transformations():
  angle_samples = 32
  epsilon = 1e-6  # or 7, 8 or higher and numerical error issues
  phi = torch.linspace(-torch.pi / 2 + epsilon, torch.pi / 2 - epsilon,
                       angle_samples).double()
  theta = torch.linspace(0 + epsilon, torch.pi - epsilon,
                         angle_samples).double()
  s_real, s_imag = spherical_riemann_to_complex(theta, phi)
  _, _ = complex_to_spherical_riemann(s_real, s_imag)

  # Inf point
  s_real = torch.Tensor([torch.inf]).double()
  s_imag = torch.Tensor([torch.inf]).double()
  _, _ = complex_to_spherical_riemann(s_real, s_imag)

  # # Zero point
  s_real = torch.Tensor([0]).double()
  s_imag = torch.Tensor([0]).double()
  _, _ = complex_to_spherical_riemann(s_real, s_imag)
