###########################
# Neural Laplace: Learning diverse classes of differential equations in the Laplace domain
# Author: Samuel Holt
###########################
# Example showing how to use all the inverse Laplace Transform algorithms individually, with a known Laplace representation F(s), and time points to evaluate for.
import argparse
import logging
from pathlib import Path
from time import strftime, time

import numpy as np
import torch

from torchlaplace.inverse_laplace import CME, DeHoog, FixedTablot, Fourier, Stehfest

device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
    file_name = Path(__file__).stem
    path_run_name = f"{file_name}-{strftime('%Y%m%d-%H%M%S')}"
    Path("./logs").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(f"logs/{path_run_name}_log.txt"),
            logging.StreamHandler(),
        ],
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )
    logger = logging.getLogger()
    np.random.seed(999)
    torch.random.manual_seed(0)

    parser = argparse.ArgumentParser("Inverse Laplace Transform Algorithms Demo")
    parser.add_argument("--time_points_to_reconstruct", type=int, default=1000)
    parser.add_argument(
        "--s_recon_terms", type=int, default=33
    )  # (ANGLE_SAMPLES * 2 + 1)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(
        "cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu"
    )
    logger.info(f"Using {device} device")

    t = torch.linspace(0.0001, 10.0, args.time_points_to_reconstruct).to(device)

    # Cosine
    def fs(so):
        return so / (so**2 + 1)  # Laplace solution

    def ft(t):
        return torch.cos(t)  # Time solution

    logger.info("")

    # Tablot

    # Evaluate s points per time input (Default, as more accurate inversion)

    decoder = FixedTablot(ilt_reconstruction_terms=args.s_recon_terms).to(device)
    t0 = time()
    f_hat_t = decoder(fs, t)
    logger.info(
        "FixedTablot Loss:\t{}\t\t| time: {}".format(
            np.sqrt(torch.nn.MSELoss()(ft(t), f_hat_t).cpu().numpy()), time() - t0
        )
    )

    # Split evaluation of s points out from that of the line integral (should be the exact same result as above)
    decoder = FixedTablot(ilt_reconstruction_terms=args.s_recon_terms).to(device)
    t0 = time()
    s, _ = decoder.compute_s(t)
    fh = fs(s)
    f_hat_t = decoder.line_integrate(fh, t)
    logger.info(
        "FixedTablot Loss (Split apart):\t{}\t| time: {}".format(
            np.sqrt(torch.nn.MSELoss()(ft(t), f_hat_t).cpu().numpy()), time() - t0
        )
    )

    decoder = FixedTablot(ilt_reconstruction_terms=args.s_recon_terms).to(device)
    t0 = time()
    s, _ = decoder.compute_s(t, time_max=torch.max(t))
    fh = fs(s)
    f_hat_t = decoder.line_integrate(fh, t, time_max=t.max().item())
    logger.info(
        "FixedTablot Loss (Split apart, Fixed Max Time):\t{}\t| time: {}".format(
            np.sqrt(torch.nn.MSELoss()(ft(t), f_hat_t).cpu().numpy()), time() - t0
        )
    )

    # Evaluate s points for one fixed time, maximum time (Less accurate, maybe more stable ?)

    decoder = FixedTablot(ilt_reconstruction_terms=args.s_recon_terms).to(device)
    t0 = time()
    f_hat_t = decoder(fs, t, time_max=torch.max(t))
    logger.info(
        "FixedTablot Loss (Fixed Max Time):\t{}\t| time: {}".format(
            np.sqrt(torch.nn.MSELoss()(ft(t), f_hat_t).cpu().numpy()), time() - t0
        )
    )

    # Stehfest - Increasing degree here, introduces numerical error that increases larger than other methods, therefore for high degree becomes unstable

    decoder = Stehfest(ilt_reconstruction_terms=args.s_recon_terms).to(device)
    t0 = time()
    f_hat_t = decoder(fs, t)
    logger.info(
        "Stehfest Loss:\t{}\t| time: {}".format(
            np.sqrt(torch.nn.MSELoss()(ft(t), f_hat_t).cpu().numpy()), time() - t0
        )
    )

    decoder = Stehfest(ilt_reconstruction_terms=args.s_recon_terms).to(device)
    t0 = time()
    s = decoder.compute_s(t)
    fh = fs(s)
    f_hat_t = decoder.line_integrate(fh, t)
    logger.info(
        "Stehfest Loss (Split apart):\t{}\t| time: {}".format(
            np.sqrt(torch.nn.MSELoss()(ft(t), f_hat_t).cpu().numpy()), time() - t0
        )
    )

    # Fourier (Un accelerated DeHoog)
    decoder = Fourier(ilt_reconstruction_terms=args.s_recon_terms).to(device)
    t0 = time()
    f_hat_t = decoder(fs, t)
    logger.info(
        "Fourier (Un accelerated DeHoog) Loss:\t{}\t| time: {}".format(
            np.sqrt(torch.nn.MSELoss()(ft(t), f_hat_t).cpu().numpy()), time() - t0
        )
    )

    decoder = Fourier(ilt_reconstruction_terms=args.s_recon_terms).to(device)
    t0 = time()
    s, T = decoder.compute_s(t)
    fh = fs(s)
    f_hat_t = decoder.line_integrate(fh, t, T)
    logger.info(
        "Fourier (Un accelerated DeHoog) Loss (Split apart):\t{}\t| time: {}".format(
            np.sqrt(torch.nn.MSELoss()(ft(t), f_hat_t).cpu().numpy()), time() - t0
        )
    )

    # DeHoog

    decoder = DeHoog(ilt_reconstruction_terms=args.s_recon_terms).to(device)
    t0 = time()
    f_hat_t = decoder(fs, t)
    logger.info(
        "DeHoog Loss:\t{}\t| time: {}".format(
            np.sqrt(torch.nn.MSELoss()(ft(t), f_hat_t).cpu().numpy()), time() - t0
        )
    )

    # Split evaluation of s points out from that of the line integral (should be the exact same result as above)
    decoder = DeHoog(ilt_reconstruction_terms=args.s_recon_terms).to(device)
    t0 = time()
    s, T = decoder.compute_s(t)
    fh = fs(s)
    f_hat_t = decoder.line_integrate(fh, t, T)
    logger.info(
        "DeHoog Loss (Split apart):\t{}\t| time: {}".format(
            np.sqrt(torch.nn.MSELoss()(ft(t), f_hat_t).cpu().numpy()), time() - t0
        )
    )

    # Single line integral
    decoder = DeHoog(ilt_reconstruction_terms=args.s_recon_terms).to(device)
    t0 = time()
    s = decoder.compute_fixed_s(torch.max(t))
    fh = fs(s)
    f_hat_t = decoder.fixed_line_integrate(fh, t, torch.max(t))
    logger.info(
        "DeHoog Loss (Fixed Line Integrate):\t{}\t| time: {}".format(
            np.sqrt(torch.nn.MSELoss()(ft(t), f_hat_t).cpu().numpy()), time() - t0
        )
    )

    decoder = DeHoog(ilt_reconstruction_terms=args.s_recon_terms).to(device)
    t0 = time()
    f_hat_t = decoder(fs, t, time_max=torch.max(t))
    logger.info(
        "DeHoog Loss (Fixed Max Time):\t{}\t| time: {}".format(
            np.sqrt(torch.nn.MSELoss()(ft(t), f_hat_t).cpu().numpy()), time() - t0
        )
    )

    # CME
    decoder = CME(ilt_reconstruction_terms=args.s_recon_terms).to(device)
    t0 = time()
    f_hat_t = decoder(fs, t)
    logger.info(
        "CME Loss:\t{}\t| time: {}".format(
            np.sqrt(torch.nn.MSELoss()(ft(t), f_hat_t).cpu().numpy()), time() - t0
        )
    )

    decoder = CME(ilt_reconstruction_terms=args.s_recon_terms).to(device)
    t0 = time()
    s, T = decoder.compute_s(t)
    fh = fs(s)
    f_hat_t = decoder.line_integrate(fh, t, T)
    logger.info(
        "CME Loss (Split apart):\t{}\t| time: {}".format(
            np.sqrt(torch.nn.MSELoss()(ft(t), f_hat_t).cpu().numpy()), time() - t0
        )
    )
    logger.info("Done")
