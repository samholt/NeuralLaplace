import torch

from torchlaplace.inverse_laplace import CME, DeHoog, FixedTablot, Fourier, Stehfest

device = torch.device("cuda:" + str(0) if torch.cuda.is_available() else "cpu")


def test_all_ilts():
    s_recon_terms = 33
    t = torch.linspace(0.0001, 10.0, 1000).to(device)

    # Cosine
    def fs(so):
        return so / (so**2 + 1)  # Laplace solution

    def ft(t):
        return torch.cos(t)  # Time solution

    # Tablot

    # Evaluate s points per time input (Default, as more accurate inversion)

    decoder = FixedTablot(ilt_reconstruction_terms=s_recon_terms).to(device)
    decoder(fs, t)

    # Split evaluation of s points out from that of the line integral (should be the exact same result as above)
    decoder = FixedTablot(ilt_reconstruction_terms=s_recon_terms).to(device)
    s, _ = decoder.compute_s(t)
    fh = fs(s)
    decoder.line_integrate(fh, t)

    decoder = FixedTablot(ilt_reconstruction_terms=s_recon_terms).to(device)
    s, _ = decoder.compute_s(t, time_max=torch.max(t))
    fh = fs(s)
    decoder.line_integrate(fh, t, time_max=t.max().item())

    # Evaluate s points for one fixed time, maximum time (Less accurate, maybe more stable ?)

    decoder = FixedTablot(ilt_reconstruction_terms=s_recon_terms).to(device)
    decoder(fs, t, time_max=torch.max(t))

    # Stehfest - Increasing degree here, introduces numerical error that increases larger than other methods, therefore for high degree becomes unstable

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
    s, T = decoder.compute_s(t)
    fh = fs(s)
    decoder.line_integrate(fh, t, T)

    # DeHoog

    decoder = DeHoog(ilt_reconstruction_terms=s_recon_terms).to(device)
    decoder(fs, t)

    # Split evaluation of s points out from that of the line integral (should be the exact same result as above)
    decoder = DeHoog(ilt_reconstruction_terms=s_recon_terms).to(device)
    s, T = decoder.compute_s(t)
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
    s, T = decoder.compute_s(t)
    fh = fs(s)
    decoder.line_integrate(fh, T)
