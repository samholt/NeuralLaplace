# pytype: skip-file
"""
This module implements the GeneralLatentODEOfficial class for training,
validating, and testing a Latent ODE model on time series data. It includes
methods for computing losses, making predictions, encoding data, and tracking
function evaluations.
"""
import matplotlib

# Ref: [Latent ODEs for Irregularly-Sampled Time Series]
# (https://github.com/YuliaRubanova/latent_ode)
import matplotlib.pyplot
import torch
from torch import nn
from torch.distributions.normal import Normal

from .latent_ode_lib.create_latent_ode_model import create_LatentODE_model_direct
from .latent_ode_lib.utils import compute_loss_all_batches_direct

matplotlib.use("Agg")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class GeneralLatentODEOfficial(nn.Module):
  """
  A PyTorch module for a General Latent ODE model with methods for training,
  validation, testing, prediction, encoding, and tracking function evaluations.
  """

  def __init__(
      self,
      input_dim,
      classif_per_tp=False,
      n_labels=1,
      obsrv_std=0.01,
      latents=2,
      hidden_units=100,
  ):
    super(GeneralLatentODEOfficial, self).__init__()

    obsrv_std = torch.Tensor([obsrv_std]).to(DEVICE)

    z0_prior = Normal(
        torch.Tensor([0.0]).to(DEVICE),
        torch.Tensor([1.0]).to(DEVICE))

    self.model = create_LatentODE_model_direct(
        input_dim,
        z0_prior,
        obsrv_std,
        DEVICE,
        classif_per_tp=classif_per_tp,
        n_labels=n_labels,
        latents=latents,
        units=hidden_units,
        gru_units=hidden_units,
    ).to(DEVICE)

    self.latents = latents

  def _get_loss(self, dl):
    loss = compute_loss_all_batches_direct(self.model,
                                           dl,
                                           device=DEVICE,
                                           classif=0)
    return loss["loss"], loss["mse"]

  def training_step(self, batch):
    loss = self.model.compute_all_losses(batch)
    return loss["loss"]

  def validation_step(self, dlval):
    loss, mse = self._get_loss(dlval)
    return loss, mse

  def test_step(self, dltest):
    loss, mse = self._get_loss(dltest)
    return loss, mse

  def predict(self, dl):
    predictions = []
    for batch in dl:
      pred_y, _ = self.model.get_reconstruction(
          batch["tp_to_predict"],
          batch["observed_data"],
          batch["observed_tp"],
          mask=batch["observed_mask"],
          n_traj_samples=1,
          mode=batch["mode"],
      )
      predictions.append(pred_y.squeeze(0))
    return torch.cat(predictions, 0)

  def encode(self, dl):
    encodings = []
    for batch in dl:
      mask = batch["observed_mask"]
      truth_w_mask = batch["observed_data"]
      if mask is not None:
        truth_w_mask = torch.cat((batch["observed_data"], mask), -1)
      mean, _ = self.model.encoder_z0(truth_w_mask,
                                        torch.flatten(batch["observed_tp"]),
                                        run_backwards=True)
      encodings.append(mean.view(-1, self.latents))
    return torch.cat(encodings, 0)

  def get_and_reset_nfes(self):
    """Returns and resets the number of function evaluations for model."""
    iteration_nfes = (self.model.encoder_z0.z0_diffeq_solver.ode_func.nfe +
                      self.model.diffeq_solver.ode_func.nfe)
    self.model.encoder_z0.z0_diffeq_solver.ode_func.nfe = 0
    self.model.diffeq_solver.ode_func.nfe = 0
    return iteration_nfes
