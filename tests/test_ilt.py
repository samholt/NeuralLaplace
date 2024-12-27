"""
This module contains tests for various Inverse Laplace Transform (ILT) methods
implemented in the torchlaplace library. The tests are designed to verify the
correctness and stability of the ILT methods by comparing their outputs against
known time-domain solutions.

The following ILT methods are tested:
- FixedTablot
- Stehfest
- Fourier
- DeHoog
- CME

Each method is tested with a cosine function, whose Laplace transform is known.
The tests include:
- Evaluating the ILT at multiple time points.
- Splitting the evaluation of s points from the line integral.
- Evaluating the ILT at a fixed maximum time.
- Testing the stability and accuracy of each method with varying reconstruction
  terms.

The tests are executed on a CUDA-enabled GPU if available, otherwise on the CPU.
"""
import torch

from torchlaplace.inverse_laplace import CME, DeHoog, FixedTablot, Fourier, Stehfest

device = torch.device("cuda:" + str(0) if torch.cuda.is_available() else "cpu")


def test_all_ilts():
  s_recon_terms = 33
  t = torch.linspace(0.0001, 10.0, 1000).to(device)

  # Cosine
  def fs(so):
    """
    Laplace transform of cos(t) is s / (s^2 + 1)
    """
    return so / (so**2 + 1)  # Laplace solution

  # Tablot
  # Evaluate s points per time input (Default, as more accurate inversion)

  decoder = FixedTablot(ilt_reconstruction_terms=s_recon_terms).to(device)
  decoder(fs, t)

  # Split evaluation of s points out from that of the line integral (should be
  # the exact same result as above)
  decoder = FixedTablot(ilt_reconstruction_terms=s_recon_terms).to(device)
  s, _ = decoder.compute_s(t)
  fh = fs(s)
  decoder.line_integrate(fh, t)

  decoder = FixedTablot(ilt_reconstruction_terms=s_recon_terms).to(device)
  s, _ = decoder.compute_s(t, time_max=torch.max(t))
  fh = fs(s)
  decoder.line_integrate(fh, t, time_max=t.max().item())

  # Evaluate s points for one fixed time, maximum time (Less accurate, maybe
  # more stable ?)

  decoder = FixedTablot(ilt_reconstruction_terms=s_recon_terms).to(device)
  decoder(fs, t, time_max=torch.max(t))

  # Stehfest - Increasing degree here, introduces numerical error that
  # increases larger than other methods, therefore for high degree becomes
  # unstable

  decoder = Stehfest(ilt_reconstruction_terms=s_recon_terms).to(device)
  decoder(fs, t)

  decoder = Stehfest(ilt_reconstruction_terms=s_recon_terms).to(device)
  s = decoder.compute_s(t)
  fh = fs(s)
  decoder.line_integrate(fh, t)

  # Fourier (Un accelerated DeHoog)
  decoder = Fourier(ilt_reconstruction_terms=s_recon_terms).to(device)
  decoder(fs, t)

  decoder = Fourier(ilt_reconstruction_terms=s_recon_terms).to(device)
  s, T = decoder.compute_s(t) # pylint: disable=invalid-name
  fh = fs(s)
  decoder.line_integrate(fh, t, T)

  # DeHoog

  decoder = DeHoog(ilt_reconstruction_terms=s_recon_terms).to(device)
  decoder(fs, t)

  # Split evaluation of s points out from that of the line integral (should be
  # the exact same result as above)
  decoder = DeHoog(ilt_reconstruction_terms=s_recon_terms).to(device)
  s, T = decoder.compute_s(t) # pylint: disable=invalid-name
  fh = fs(s)
  decoder.line_integrate(fh, t, T)

  # Single line integral
  decoder = DeHoog(ilt_reconstruction_terms=s_recon_terms).to(device)
  s = decoder.compute_fixed_s(torch.max(t))
  fh = fs(s)
  decoder.fixed_line_integrate(fh, t, torch.max(t))

  decoder = DeHoog(ilt_reconstruction_terms=s_recon_terms).to(device)
  decoder(fs, t, time_max=torch.max(t))

  # CME
  decoder = CME(ilt_reconstruction_terms=s_recon_terms).to(device)
  decoder(fs, t)

  decoder = CME(ilt_reconstruction_terms=s_recon_terms).to(device)
  s, T = decoder.compute_s(t) # pylint: disable=invalid-name
  fh = fs(s)
  decoder.line_integrate(fh, T)
