# pytype: skip-file
"""
Neural Laplace: Learning diverse classes of differential equations in the
  Laplace domain
Author: Samuel Holt

This module demonstrates the Delay Differential Equation (DDE) of the
Lotka Volterra System with a delay. It depends on additional packages to be
installed (see setup.py or documentation to install all packages) to sample
from a DDE. Note that sampling a DDE is a slow process, therefore the dataset
function uses memoization to keep the sampled trajectories for the next run
time, if applicable. Sampling 1,000 trajectories can take approximately 15
minutes.

The script includes:
- Argument parsing for various hyperparameters and settings.
- Definition of the ReverseGRUEncoder class for encoding observed trajectories
  into latent vectors.
- Definition of the LaplaceRepresentationFunc class for representing functions
  in the Laplace domain using Riemann Sphere coordinates.
- Visualization functions to plot the results.
- Main execution block to train, validate, and test the model.
"""
###########################
# Neural Laplace: Learning diverse classes of differential equations in the
#   Laplace domain
# Author: Samuel Holt
###########################
# Delay Differential Equation of the Lotka Volterra System (with a delay)
# This file depends on the additional packages to be installed (see setup.py or
#   documentation to install all packages) to sample from a DDE
# Note sampling a DDE is a slow process, therefore the dataset function uses
#   memoization to keep the sampled trajectories for next run time, if
#   applicable. Sampling 1,000 trajectories can take approx. 15 minutes
import argparse
import logging
from copy import deepcopy
from pathlib import Path
from time import strftime, time

import numpy as np
import torch
from torch import nn

from experiments.datasets import generate_data_set
from torchlaplace import laplace_reconstruct

parser = argparse.ArgumentParser(
    "Lotka Volterra System Delay Differential Equation demo")

parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--learning_rate", type=float, default=1e-3)
parser.add_argument("--extrapolate", action="store_false")  # Default True
parser.add_argument("--epochs", type=int, default=1000)
parser.add_argument("--trajectories_to_sample", type=int, default=1000)
parser.add_argument("--time_points_to_sample", type=int, default=200)
parser.add_argument("--noise_std", type=float, default=0.0)
parser.add_argument("--normalize_dataset", action="store_false")  # Default True
parser.add_argument("--encode_obs_time", action="store_false")  # Default True
parser.add_argument("--hidden_units", type=int, default=64)
parser.add_argument("--latent_dim", type=int, default=2)
parser.add_argument("--s_recon_terms", type=int,
                    default=33)  # (ANGLE_SAMPLES * 2 + 1)
parser.add_argument("--viz_per_epoch", type=int, default=6)
parser.add_argument("--patience", nargs="?", type=int, const=500)
parser.add_argument("--viz", action="store_true")
parser.add_argument("--gpu", type=int, default=0)
args = parser.parse_args()
patience = args.patience

device = torch.device("cuda:" +
                      str(args.gpu) if torch.cuda.is_available() else "cpu")


# Model (encoder and Laplace representation func)
class ReverseGRUEncoder(nn.Module):
  """
  A GRU-based encoder that processes observed trajectories in reverse order
  and encodes them into a latent vector. Optionally includes observation times
  in the input.
  Args:
    dimension_in (int): Dimensionality of the input data.
    latent_dim (int): Dimensionality of the latent vector.
    hidden_units (int): Number of hidden units in the GRU.
    encode_obs_time (bool): Whether to include observation times in the input.
  """
  # Encodes observed trajectory into latent vector
  def __init__(self,
               dimension_in,
               latent_dim,
               hidden_units,
               encode_obs_time=True):
    super(ReverseGRUEncoder, self).__init__()
    self.encode_obs_time = encode_obs_time
    if self.encode_obs_time:
      dimension_in += 1
    self.gru = nn.GRU(dimension_in, hidden_units, 2, batch_first=True)
    self.linear_out = nn.Linear(hidden_units, latent_dim)
    nn.init.xavier_uniform_(self.linear_out.weight)

  def forward(self, observed_data, observed_time_points):
    trajectories_to_encode = (
        observed_data  # (batch_size, t_observed_dim, observed_dim)
    )
    if self.encode_obs_time:
      trajectories_to_encode = torch.cat(
          (
              observed_data,
              observed_time_points.view(1, -1, 1).repeat(observed_data.shape[0],
                                                          1, 1),
          ),
          dim=2,
      )
    reversed_trajectories_to_encode = torch.flip(trajectories_to_encode, (1,))
    out, _ = self.gru(reversed_trajectories_to_encode)
    return self.linear_out(out[:, -1, :])


class LaplaceRepresentationFunc(nn.Module):
  """
  A neural network module that represents a function for Laplace representation
  in Riemann Sphere coordinates.
  Args:
    s_dim (int): The dimension of the sphere surface reconstruction terms.
    output_dim (int): The output dimension.
    latent_dim (int): The latent encoding dimension.
    hidden_units (int, optional): The number of hidden units in the linear
      layers. Default is 64.
  Attributes:
    s_dim (int): The dimension of the sphere surface reconstruction terms.
    output_dim (int): The output dimension.
    latent_dim (int): The latent encoding dimension.
    linear_tanh_stack (nn.Sequential): A sequential container of linear and
      Tanh layers.
    phi_scale (torch.Tensor): The scaling factor for the phi angle.
  Methods:
    forward(i):
      Forward pass of the network. Takes an input tensor `i` and returns
      the theta and phi angles in Riemann Sphere coordinates.
      Args:
        i (torch.Tensor): Input tensor of shape (batch_size,
          s_dim * 2 + latent_dim).
      Returns:
        tuple: A tuple containing:
          - theta (torch.Tensor): Tensor of shape (batch_size, output_dim,
            s_dim) representing the theta angles.
          - phi (torch.Tensor): Tensor of shape (batch_size, output_dim, s_dim)
            representing the phi angles.
  """
  # SphereSurfaceModel : C^{b+k} -> C^{bxd} -
  # In Riemann Sphere Co ords : b dim s reconstruction terms, k is
  # latent encoding dimension, d is output dimension
  def __init__(self, s_dim, out_dim, latent_dim, hidden_units=64):
    super(LaplaceRepresentationFunc, self).__init__()
    self.s_dim = s_dim
    self.output_dim = out_dim
    self.latent_dim = latent_dim
    self.linear_tanh_stack = nn.Sequential(
        nn.Linear(s_dim * 2 + latent_dim, hidden_units),
        nn.Tanh(),
        nn.Linear(hidden_units, hidden_units),
        nn.Tanh(),
        nn.Linear(hidden_units, (s_dim) * 2 * output_dim),
    )

    for m in self.linear_tanh_stack.modules():
      if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)

    phi_max = torch.pi / 2.0
    self.phi_scale = phi_max - -torch.pi / 2.0

  def forward(self, i):
    out = self.linear_tanh_stack(i.view(-1,
                                        self.s_dim * 2 + self.latent_dim)).view(
                                            -1, 2 * self.output_dim, self.s_dim)
    theta = (nn.Tanh()(out[:, :self.output_dim, :]) * torch.pi
            )  # From - pi to + pi
    phi = (nn.Tanh()(out[:, self.output_dim:, :]) * self.phi_scale / 2.0 -
           torch.pi / 2.0 + self.phi_scale / 2.0)  # Form -pi / 2 to + pi / 2
    return theta, phi


def visualize(tp_to_pred, predictions_inner, data_to_predict, path_name,
              epoch_inner):
  tp_to_pred = torch.squeeze(tp_to_pred)
  predictions_inner = torch.squeeze(predictions_inner)
  y_true = torch.squeeze(data_to_predict)

  margin = 1.1
  ax_top_one.cla()
  ax_top_one.set_title("Sample 0 (Phase Portrait)")
  ax_top_one.set_xlabel("x(t)")
  ax_top_one.set_ylabel("y(t)")
  ax_top_one.plot(y_true.cpu().numpy()[0, :, 0],
                  y_true.cpu().numpy()[0, :, 1], "k--")
  ax_top_one.plot(
      predictions_inner.cpu().numpy()[0, :, 0],
      predictions_inner.cpu().numpy()[0, :, 1],
      "b-",
  )
  ax_top_one.set_xlim(
      y_true.cpu()[0, :, 0].min() * margin,
      y_true.cpu()[0, :, 0].max() * margin,
  )
  ax_top_one.set_ylim(
      y_true.cpu()[0, :, 1].min() * margin,
      y_true.cpu()[0, :, 1].max() * margin,
  )
  ax_bottom_one.cla()
  ax_bottom_one.set_title("Sample 0 (Trajectories)")
  ax_bottom_one.set_xlabel("time")
  ax_bottom_one.set_ylabel("x(t), y(t)")
  ax_bottom_one.plot(
      tp_to_pred.cpu().numpy(),
      y_true.cpu().numpy()[0, :, 0],
      "k--",
      tp_to_pred.cpu().numpy(),
      y_true.cpu().numpy()[0, :, 1],
      "k--",
  )
  ax_bottom_one.plot(
      tp_to_pred.cpu().numpy(),
      predictions_inner.cpu().numpy()[0, :, 0],
      tp_to_pred.cpu().numpy(),
      predictions_inner.cpu().numpy()[0, :, 1],
  )
  ax_bottom_one.set_xlim(tp_to_pred.cpu().min(), tp_to_pred.cpu().max())
  ax_bottom_one.set_ylim(
      y_true.cpu()[0, :, :].min() * margin,
      y_true.cpu()[0, :, :].max() * margin,
  )

  ax_top_two.cla()
  ax_top_two.set_title("Sample 1 (Phase Portrait)")
  ax_top_two.set_xlabel("x(t)")
  ax_top_two.set_ylabel("y(t)")
  ax_top_two.plot(y_true.cpu().numpy()[1, :, 0],
                  y_true.cpu().numpy()[1, :, 1], "k--")
  ax_top_two.plot(
      predictions_inner.cpu().numpy()[1, :, 0],
      predictions_inner.cpu().numpy()[1, :, 1],
      "b-",
  )
  ax_top_two.set_xlim(
      y_true.cpu()[1, :, 0].min() * margin,
      y_true.cpu()[1, :, 0].max() * margin,
  )
  ax_top_two.set_ylim(
      y_true.cpu()[1, :, 1].min() * margin,
      y_true.cpu()[1, :, 1].max() * margin,
  )
  ax_bottom_two.cla()
  ax_bottom_two.set_title("Sample 1 (Trajectories)")
  ax_bottom_two.set_xlabel("time")
  ax_bottom_two.set_ylabel("x(t), y(t)")
  ax_bottom_two.plot(
      tp_to_pred.cpu().numpy(),
      y_true.cpu().numpy()[1, :, 0],
      "k--",
      tp_to_pred.cpu().numpy(),
      y_true.cpu().numpy()[1, :, 1],
      "k--",
  )
  ax_bottom_two.plot(
      tp_to_pred.cpu().numpy(),
      predictions_inner.cpu().numpy()[1, :, 0],
      tp_to_pred.cpu().numpy(),
      predictions_inner.cpu().numpy()[1, :, 1],
  )
  ax_bottom_two.set_xlim(tp_to_pred.cpu().min(), tp_to_pred.cpu().max())
  ax_bottom_two.set_ylim(
      y_true.cpu()[1, :, :].min() * margin,
      y_true.cpu()[1, :, :].max() * margin,
  )

  ax_top_three.cla()
  ax_top_three.set_title("Sample 2 (Phase Portrait)")
  ax_top_three.set_xlabel("x(t)")
  ax_top_three.set_ylabel("y(t)")
  ax_top_three.plot(y_true.cpu().numpy()[2, :, 0],
                    y_true.cpu().numpy()[2, :, 1], "k--")
  ax_top_three.plot(
      predictions_inner.cpu().numpy()[2, :, 0],
      predictions_inner.cpu().numpy()[2, :, 1],
      "b-",
  )
  ax_top_three.set_xlim(
      y_true.cpu()[2, :, 0].min() * margin,
      y_true.cpu()[2, :, 0].max() * margin,
  )
  ax_top_three.set_ylim(
      y_true.cpu()[2, :, 1].min() * margin,
      y_true.cpu()[2, :, 1].max() * margin,
  )
  ax_bottom_three.cla()
  ax_bottom_three.set_title("Sample 2 (Trajectories)")
  ax_bottom_three.set_xlabel("time")
  ax_bottom_three.set_ylabel("x(t), y(t)")
  ax_bottom_three.plot(
      tp_to_pred.cpu().numpy(),
      y_true.cpu().numpy()[2, :, 0],
      "k--",
      tp_to_pred.cpu().numpy(),
      y_true.cpu().numpy()[2, :, 1],
      "k--",
  )
  ax_bottom_three.plot(
      tp_to_pred.cpu().numpy(),
      predictions_inner.cpu().numpy()[2, :, 0],
      tp_to_pred.cpu().numpy(),
      predictions_inner.cpu().numpy()[2, :, 1],
  )
  ax_bottom_three.set_xlim(tp_to_pred.cpu().min(), tp_to_pred.cpu().max())
  ax_bottom_three.set_ylim(
      y_true.cpu()[2, :, :].min() * margin,
      y_true.cpu()[2, :, :].max() * margin,
  )

  fig.tight_layout()
  plt.savefig(f"png/{path_name}-{epoch_inner:03d}")
  plt.draw()
  plt.pause(0.001)


np.random.seed(999)
file_name = Path(__file__).stem

if __name__ == "__main__":
  path_run_name = f"{file_name}-{strftime('%Y%m%d-%H%M%S')}"

  Path("./logs").mkdir(parents=True, exist_ok=True)
  if args.viz:
    Path("./png").mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt

    plt.style.use("tableau-colorblind10")
    plt.rcParams.update({"font.size": 12})
    fig = plt.figure(figsize=(12, 8), facecolor="white")
    ax_top_one = fig.add_subplot(231, frameon=False)
    ax_top_two = fig.add_subplot(232, frameon=False)
    ax_top_three = fig.add_subplot(233, frameon=False)
    ax_bottom_one = fig.add_subplot(234, frameon=False)
    ax_bottom_two = fig.add_subplot(235, frameon=False)
    ax_bottom_three = fig.add_subplot(236, frameon=False)
    plt.show(block=False)

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

  logger.info("Using %s device", device)

  torch.random.manual_seed(0)

  (
      input_dim,
      output_dim,
      dltrain,
      dlval,
      dltest,
      _,
      _,
      _,
  ) = generate_data_set(
      "lotka_volterra_system_with_delay",
      device,
      trajectories_to_sample=args.trajectories_to_sample,
      extrap=args.extrapolate,
      normalize=args.normalize_dataset,
      noise_std=args.noise_std,
      t_nsamples=args.time_points_to_sample,
      observe_step=1,
      predict_step=1,
  )

  if not patience:
    patience = args.epochs

  # Model
  encoder = ReverseGRUEncoder(
      input_dim,
      args.latent_dim,
      args.hidden_units // 2,
      encode_obs_time=args.encode_obs_time,
  ).to(device)
  laplace_rep_func = LaplaceRepresentationFunc(args.s_recon_terms, output_dim,
                                               args.latent_dim).to(device)
  params = list(laplace_rep_func.parameters()) + list(encoder.parameters())
  optimizer = torch.optim.Adam(params, lr=args.learning_rate)
  loss_fn = torch.nn.MSELoss()

  best_loss = float("inf")
  waiting = 0

  for epoch in range(args.epochs):
    iteration = 0
    epoch_train_loss_it_cum = 0
    start_time = time()
    laplace_rep_func.train(), encoder.train() # pylint: disable=expression-not-assigned
    for batch in dltrain:
      optimizer.zero_grad()
      trajs_to_encode = batch[
          "observed_data"]  # (batch_size, t_observed_dim, observed_dim)
      observed_tp = batch["observed_tp"]  # (1, t_observed_dim)
      p = encoder(
          trajs_to_encode,
          observed_tp)  # p is the latent tensor encoding the initial states
      tp_to_predict = batch["tp_to_predict"]
      predictions = laplace_reconstruct(laplace_rep_func,
                                        p,
                                        tp_to_predict,
                                        recon_dim=output_dim)
      loss = loss_fn(
          torch.flatten(predictions),
          torch.flatten(batch["data_to_predict"]),
      )
      loss.backward()
      torch.nn.utils.clip_grad_norm_(params, 1)
      optimizer.step()
      epoch_train_loss_it_cum += loss.item()
      iteration += 1
    epoch_train_loss = epoch_train_loss_it_cum / iteration
    epoch_duration = time() - start_time

    # Validation step
    laplace_rep_func.eval(), encoder.eval() # pylint: disable=expression-not-assigned
    cum_val_loss = 0
    cum_val_batches = 0
    for batch in dlval:
      trajs_to_encode = batch[
          "observed_data"]  # (batch_size, t_observed_dim, observed_dim)
      observed_tp = batch["observed_tp"]  # (1, t_observed_dim)
      p = encoder(
          trajs_to_encode,
          observed_tp)  # p is the latent tensor encoding the initial states
      tp_to_predict = batch["tp_to_predict"]
      predictions = laplace_reconstruct(laplace_rep_func,
                                        p,
                                        tp_to_predict,
                                        recon_dim=output_dim)
      cum_val_loss += loss_fn(
          torch.flatten(predictions),
          torch.flatten(batch["data_to_predict"]),
      ).item()
      cum_val_batches += 1
    if (epoch % args.viz_per_epoch == 0) and args.viz:
      visualize(
          tp_to_predict.detach(),
          predictions.detach(),
          batch["data_to_predict"].detach(),
          path_run_name,
          epoch,
      )
    val_mse = cum_val_loss / cum_val_batches
    logger.info(
        "[epoch=%d] epoch_duration=%.2f | train_loss=%f\t| val_mse=%f\t|",
        epoch, epoch_duration, epoch_train_loss, val_mse)

    # Early stopping procedure
    if val_mse < best_loss:
      best_loss = val_mse
      best_laplace_rep_func = deepcopy(laplace_rep_func.state_dict())
      best_encoder = deepcopy(encoder.state_dict())
      waiting = 0
    elif waiting > patience:
      break
    else:
      waiting += 1

  # Load best model
  laplace_rep_func.load_state_dict(best_laplace_rep_func)
  encoder.load_state_dict(best_encoder)

  # Test step
  laplace_rep_func.eval(), encoder.eval() # pylint: disable=expression-not-assigned
  cum_test_loss = 0
  cum_test_batches = 0
  for batch in dltest:
    trajs_to_encode = batch[
        "observed_data"]  # (batch_size, t_observed_dim, observed_dim)
    observed_tp = batch["observed_tp"]  # (1, t_observed_dim)
    p = encoder(
        trajs_to_encode,
        observed_tp)  # p is the latent tensor encoding the initial states
    tp_to_predict = batch["tp_to_predict"]
    predictions = laplace_reconstruct(laplace_rep_func, p, tp_to_predict)
    cum_test_loss += loss_fn(torch.flatten(predictions),
                             torch.flatten(batch["data_to_predict"])).item()
    cum_test_batches += 1
  test_mse = cum_test_loss / cum_test_batches
  logger.info("test_mse= %f", test_mse)
