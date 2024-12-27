# pytype: skip-file
"""
This module contains implementations of various neural network models for
Ordinary Differential Equations (ODEs) and Latent ODEs, including ODE,
GeneralLatentODE, SolverWrapper, and VAE_Baseline. It also includes utility
functions and classes for creating and training these models.
"""
# Ref: [Neural Flows: Efficient Alternative to Neural ODEs]
# (https://github.com/mbilos/neural-flows-experiments)
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import kl_divergence
from torch.distributions.normal import Normal
from torch.nn import Module

from .data_utils import (
    check_mask,
    compute_binary_CE_loss,
    compute_loss_all_batches,
    compute_mse,
    compute_multiclass_CE_loss,
    compute_poisson_proc_likelihood,
    init_network_weights,
    masked_gaussian_log_density,
    sample_standard_gaussian,
)
from .flow import CouplingFlow, ResNetFlow
from .gru import GRUFlow
from .ode import ODEModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logger = logging.getLogger()


class ODE(Module):
  """
  ODE model class.
  Args:
    dim (int): Dimensionality of the input.
    obsrv_std (float): Observation standard deviation.
    n_classes (int): Number of classes.
    model (str): Type of model ('ode' or 'flow').
    flow_model (str): Type of flow model ('coupling' or 'resnet').
    latents (int): Number of latent dimensions.
    hidden_dim (int): Dimension of hidden layers.
    hidden_layers (int): Number of hidden layers.
    solver (str): Solver type.
    odenet (str): ODE network type.
    activation (str): Activation function.
    final_activation (str): Final activation function.
    solver_step (float): Solver step size.
    atol (float): Absolute tolerance.
    rtol (float): Relative tolerance.
    flow_layers (int): Number of flow layers.
    time_net (str): Time network type.
    time_hidden_dim (int): Dimension of time hidden layers.
  Methods:
    _get_loss(batch): Computes loss for a batch.
    _get_loss_on_dl(dl): Computes loss on a dataloader.
    training_step(batch): Training step.
    validation_step(dlval): Validation step.
    test_step(dltest): Test step.
    _sample_trajectories(path): Samples trajectories and saves to path.
    predict(dl): Predicts using the model.
  """
  def __init__(
      self,
      dim,
      obsrv_std=0.001, # pylint: disable=unused-argument
      n_classes=1, # pylint: disable=unused-argument
      model="ode",
      flow_model="coupling",
      latents=1, # pylint: disable=unused-argument
      hidden_dim=72,
      hidden_layers=2,
      solver="euler",
      odenet="concat",
      activation="ELU",
      final_activation="Tanh",
      solver_step=0.05,
      atol=0.0001,
      rtol=0.001,
      flow_layers=1,
      time_net="TimeLinear",
      time_hidden_dim=1,
  ):
    super(ODE, self).__init__()
    if model == "ode":
      self.model = ODEModel(
          dim,
          odenet,
          [hidden_dim] * hidden_layers,
          activation,
          final_activation,
          solver,
          solver_step,
          atol,
          rtol,
      )
    elif model == "flow":
      if flow_model == "coupling":
        self.model = CouplingFlow(
            dim,
            flow_layers,
            [hidden_dim] * hidden_layers,
            time_net,
            time_hidden_dim,
        )
      elif flow_model == "resnet":
        self.model = ResNetFlow(
            dim,
            flow_layers,
            [hidden_dim] * hidden_layers,
            time_net,
            time_hidden_dim,
        )

  def _get_loss(self, batch):
    y = self.model(batch["observed_data"][:, 0, :], batch["observed_tp"])
    assert y.shape == batch["data_to_predict"].shape
    loss = torch.mean((y - batch["data_to_predict"])**2)
    return loss

  def _get_loss_on_dl(self, dl):
    losses = []
    for batch in dl:
      losses.append(self._get_loss(batch).detach())
    return torch.mean(torch.stack(losses))

  def training_step(self, batch):
    return self._get_loss(batch)

  def validation_step(self, dlval):
    mse = self._get_loss_on_dl(dlval)
    return mse, mse

  def test_step(self, dltest):
    mse = self._get_loss_on_dl(dltest)
    return mse, mse

  def _sample_trajectories(self, path):
    N, M, T = 21, 200, 30 # pylint: disable=invalid-name
    x = torch.linspace(-5, 5, N).view(N, 1, 1)
    t = torch.linspace(0, T, M).view(1, M, 1).repeat(N, 1, 1)
    y = self.model(x, t)
    np.savez(
        path,
        x=x.detach().cpu().numpy(),
        t=t.detach().cpu().numpy(),
        y=y.detach().cpu().numpy(),
    )

  def predict(self, dl):
    predictions = []
    for batch in dl:
      predictions.append(
          self.model(batch["observed_data"], batch["tp_to_predict"]))
    return torch.cat(predictions, 0)


class GeneralLatentODE(Module):
  """
  GeneralLatentODE is a neural network model for learning latent Ordinary
    Differential Equations (ODEs).

  Args:
    dim (int): Dimensionality of the input data.
    obsrv_std (float, optional): Standard deviation of the observation noise.
      Default is 0.001.
    n_classes (int, optional): Number of classes for classification tasks.
      Default is 1.
    model (str, optional): Type of model to use. Default is "ode".
    flow_model (str, optional): Type of flow model to use. Default is
      "coupling".
    latents (int, optional): Number of latent dimensions. Default is 2.
    hidden_dim (int, optional): Dimensionality of hidden layers. Default is 72.
    hidden_layers (int, optional): Number of hidden layers. Default is 2.
    solver (str, optional): Solver to use for ODE integration. Default is
      "euler".
  """

  def __init__(
      self,
      dim,
      obsrv_std=0.001,
      n_classes=1,
      model="ode",
      flow_model="coupling",
      latents=2,
      hidden_dim=72,
      hidden_layers=2,
      solver="euler",
  ):
    super(GeneralLatentODE, self).__init__()

    z0_prior = torch.distributions.Normal(
        torch.Tensor([0.0]).to(DEVICE),
        torch.Tensor([1.0]).to(DEVICE))

    obsrv_std = torch.Tensor([obsrv_std]).to(DEVICE)

    self.model = create_LatentODE_model(
        dim,
        z0_prior,
        obsrv_std,
        DEVICE,
        n_labels=n_classes,
        model=model,
        flow_model=flow_model,
        latents=latents,
        rec_dims=hidden_dim,
        hidden_dim=hidden_dim,
        hidden_layers=hidden_layers,
        solver=solver,
    )

    self.latents = latents

  def _get_loss(self, dl):
    loss = compute_loss_all_batches(self.model,
                                    dl,
                                    classify=0,
                                    data="",
                                    device=DEVICE)
    return loss["loss"], loss["mse"], loss["acc"]

  def training_step(self, batch):
    loss = self.model.compute_all_losses(batch)
    return loss["loss"]

  def validation_step(self, dlval):
    loss, mse, _ = self._get_loss(dlval)
    return loss, mse

  def test_step(self, dltest):
    loss, mse, _ = self._get_loss(dltest)
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
                                        batch["observed_tp"],
                                        run_backwards=True)
      encodings.append(mean.view(-1, self.latents))
    return torch.cat(encodings, 0)

  def get_and_reset_nfes(self):
    """Returns and resets the number of function evaluations for model."""
    iteration_nfes = (self.model.encoder_z0.z0_diffeq_solver.solver.nfe +
                      self.model.diffeq_solver.solver.nfe)
    self.model.encoder_z0.z0_diffeq_solver.solver.nfe = 0
    self.model.diffeq_solver.solver.nfe = 0
    return iteration_nfes


class SolverWrapper(nn.Module):
  """
  A wrapper for a solver module to handle input and time dimensions.
  """

  def __init__(self, solver):
    super().__init__()
    self.solver = solver

  def forward(self, x, t, backwards=False): # pylint: disable=unused-argument
    assert len(x.shape) - len(t.shape) == 1
    t = t.unsqueeze(-1)
    if t.shape[-3] != x.shape[-3]:
      t = t.repeat_interleave(x.shape[-3], dim=-3)
    if len(x.shape) == 4:
      t = t.repeat_interleave(x.shape[0], dim=0)
    y = self.solver(x, t)  # (1, batch_size, times, dim)
    return y


def create_LatentODE_model( # pylint: disable=invalid-name
    input_dim,
    z0_prior,
    obsrv_std,
    device, # pylint: disable=unused-argument
    classif_per_tp=False,
    n_labels=1,
    data="hopper",
    latents=2,
    rec_dims=100,
    hidden_dim=100,
    hidden_layers=3,
    model="ode",
    odenet="concat",
    activation="Tanh",
    final_activation="Identity",
    solver="euler",
    solver_step=0.05,
    atol=0.0001,
    rtol=0.001,
    flow_model="coupling",
    flow_layers=2,
    time_net="TimeLinear",
    time_hidden_dim=8,
    gru_units=50,
    classify=0,
):
  classif_per_tp = data == "activity"

  z0_diffeq_solver = None
  n_rec_dims = rec_dims
  enc_input_dim = int(input_dim) * 2  # we concatenate the mask
  gen_data_dim = input_dim

  z0_dim = latents
  hidden_dims = [hidden_dim] * hidden_layers

  if model == "ode":
    z0_diffeq_solver = SolverWrapper(
        ODEModel(
            n_rec_dims,
            odenet,
            hidden_dims,
            activation,
            final_activation,
            solver,
            solver_step,
            atol,
            rtol,
        ))
    diffeq_solver = SolverWrapper(
        ODEModel(
            latents,
            odenet,
            hidden_dims,
            activation,
            final_activation,
            solver,
            solver_step,
            atol,
            rtol,
        ))
  elif model == "flow":
    if flow_model == "coupling":
      flow = CouplingFlow
    elif flow_model == "resnet":
      flow = ResNetFlow
    elif flow_model == "gru":
      flow = GRUFlow
    else:
      raise ValueError("Unknown flow transformation")

    z0_diffeq_solver = SolverWrapper(
        flow(n_rec_dims,
             flow_layers,
             hidden_dims,
             time_net,
             time_hidden_dim)) # pytype: disable=wrong-arg-count
    diffeq_solver = SolverWrapper(
        flow(latents,
             flow_layers,
             hidden_dims,
             time_net,
             time_hidden_dim)) # pytype: disable=wrong-arg-count
  else:
    raise NotImplementedError

  encoder_z0 = Encoder_z0_ODE_RNN(
      n_rec_dims,
      enc_input_dim,
      z0_diffeq_solver,
      z0_dim=z0_dim,
      n_gru_units=gru_units,
      device=DEVICE,
  ).to(DEVICE)

  decoder = Decoder(latents, gen_data_dim).to(DEVICE)

  return LatentODE(
      input_dim=gen_data_dim,
      latent_dim=latents,
      encoder_z0=encoder_z0,
      decoder=decoder,
      diffeq_solver=diffeq_solver,
      z0_prior=z0_prior,
      device=DEVICE,
      obsrv_std=obsrv_std,
      use_poisson_proc=False,
      use_binary_classif=classify,
      linear_classifier=False,
      classif_per_tp=classif_per_tp,
      n_labels=n_labels,
      train_classif_w_reconstr=(data == "physionet" or data == "activity"),
  ).to(DEVICE)


def get_mask(x):
  x = x.unsqueeze(0)
  n_data_dims = x.size(-1) // 2
  mask = x[:, :, n_data_dims:]
  check_mask(x[:, :, :n_data_dims], mask)
  mask = (torch.sum(mask, -1, keepdim=True) > 0).float()
  assert not torch.isnan(mask).any()
  return mask.squeeze(0)


class Encoder_z0_ODE_RNN(nn.Module): # pylint: disable=invalid-name
  """
  Encoder_z0_ODE_RNN is a neural network module that encodes input sequences
  into latent space using an ODE-RNN approach.
  """

  def __init__(
      self,
      latent_dim,
      input_dim,
      z0_diffeq_solver=None,
      z0_dim=None,
      n_gru_units=100, # pylint: disable=unused-argument
      device=torch.device("cpu"),
  ):
    super().__init__()

    if z0_dim is None:
      self.z0_dim = latent_dim
    else:
      self.z0_dim = z0_dim

    self.lstm = nn.LSTMCell(input_dim, latent_dim)

    self.z0_diffeq_solver = z0_diffeq_solver
    self.latent_dim = latent_dim
    self.input_dim = input_dim
    self.device = device
    self.extra_info = None

    self.transform_z0 = nn.Sequential(
        nn.Linear(latent_dim, 100),
        nn.Tanh(),
        nn.Linear(100, self.z0_dim * 2),
    )
    init_network_weights(self.transform_z0)

  def forward(self, data, time_steps, run_backwards=True, save_info=False): # pylint: disable=unused-argument
    assert not torch.isnan(data).any()
    assert not torch.isnan(time_steps).any()

    n_traj, _, _ = data.size()
    latent = self.run_odernn(data, time_steps, run_backwards)

    latent = latent.reshape(1, n_traj, self.latent_dim)

    mean_z0, std_z0 = self.transform_z0(latent).chunk(2, dim=-1)
    std_z0 = F.softplus(std_z0) # pylint: disable=not-callable

    return mean_z0, std_z0

  def run_odernn(self, data, time_steps, run_backwards=True):
    batch_size, _, _ = data.size()
    prev_t, t_i = time_steps[:, -1] + 0.01, time_steps[:, -1]

    time_points_iter = range(0, time_steps.shape[1])
    if run_backwards:
      time_points_iter = reversed(time_points_iter)

    h = torch.zeros(batch_size, self.latent_dim).to(data)
    c = torch.zeros(batch_size, self.latent_dim).to(data)

    for i in time_points_iter:
      t = (t_i - prev_t).unsqueeze(1)
      h = self.z0_diffeq_solver(h.unsqueeze(1), t).squeeze(1)

      xi = data[:, i, :]
      h_, c_ = self.lstm(xi, (h, c))
      mask = get_mask(xi)

      h = mask * h_ + (1 - mask) * h
      c = mask * c_ + (1 - mask) * c

      prev_t, t_i = time_steps[:, i], time_steps[:, i - 1]

    return h


class Decoder(nn.Module):
  """
  Decoder class for transforming latent space representations back to input
  space.
  """

  def __init__(self, latent_dim, input_dim):
    super().__init__()
    decoder = nn.Sequential(nn.Linear(latent_dim, input_dim),)
    init_network_weights(decoder)
    self.decoder = decoder

  def forward(self, data):
    return self.decoder(data)


class VAE_Baseline(nn.Module): # pylint: disable=invalid-name
  """
  Variational Autoencoder (VAE) Baseline model for time series data.

  Args:
    input_dim (int): Dimensionality of the input data.
    latent_dim (int): Dimensionality of the latent space.
    z0_prior (torch.distributions.Distribution): Prior distribution for the
      initial latent state.
    device (torch.device): Device to run the model on.
    obsrv_std (float, optional): Standard deviation for the observation noise.
      Default is 0.01.
    use_binary_classif (bool, optional): Whether to use binary classification.
      Default is False.
    classif_per_tp (bool, optional): Whether to classify per time point.
      Default is False.
    use_poisson_proc (bool, optional): Whether to use Poisson process
      likelihood. Default is False.
    linear_classifier (bool, optional): Whether to use a linear classifier.
      Default is False.
    n_labels (int, optional): Number of labels for classification. Default is 1.
    train_classif_w_reconstr (bool, optional): Whether to train classification
    with reconstruction. Default is False.
  """

  def __init__(
      self,
      input_dim,
      latent_dim,
      z0_prior,
      device,
      obsrv_std=0.01,
      use_binary_classif=False,
      classif_per_tp=False,
      use_poisson_proc=False,
      linear_classifier=False,
      n_labels=1,
      train_classif_w_reconstr=False,
  ):

    super(VAE_Baseline, self).__init__()

    self.input_dim = input_dim
    self.latent_dim = latent_dim
    self.device = device
    self.n_labels = n_labels

    self.obsrv_std = torch.Tensor([obsrv_std]).to(device)

    self.z0_prior = z0_prior
    self.use_binary_classif = use_binary_classif
    self.classif_per_tp = classif_per_tp
    self.use_poisson_proc = use_poisson_proc
    self.linear_classifier = linear_classifier
    self.train_classif_w_reconstr = train_classif_w_reconstr

    z0_dim = latent_dim
    if use_poisson_proc:
      z0_dim += latent_dim

    if use_binary_classif:
      if linear_classifier:
        self.classifier = nn.Sequential(nn.Linear(z0_dim, n_labels))
      else:
        self.classifier = create_classifier(z0_dim, n_labels)
      init_network_weights(self.classifier)

  def get_gaussian_likelihood(self, truth, pred_y, mask=None):
    # pred_y shape [n_traj_samples, n_traj, n_tp, n_dim]
    # truth shape  [n_traj, n_tp, n_dim]

    # Compute likelihood of the data under the predictions
    truth_repeated = truth.repeat(pred_y.size(0), 1, 1, 1)

    if mask is not None:
      mask = mask.repeat(pred_y.size(0), 1, 1, 1)
    log_density_data = masked_gaussian_log_density(pred_y,
                                                   truth_repeated,
                                                   obsrv_std=self.obsrv_std,
                                                   mask=mask)
    log_density_data = log_density_data.permute(1, 0)
    log_density = torch.mean(log_density_data, 1)

    # shape: [n_traj_samples]
    return log_density

  def get_mse(self, truth, pred_y, mask=None):
    # pred_y shape [n_traj_samples, n_traj, n_tp, n_dim]
    # truth shape  [n_traj, n_tp, n_dim]

    # Compute likelihood of the data under the predictions
    truth_repeated = truth.repeat(pred_y.size(0), 1, 1, 1)

    if mask is not None:
      mask = mask.repeat(pred_y.size(0), 1, 1, 1)

    # Compute likelihood of the data under the predictions
    log_density_data = compute_mse(pred_y, truth_repeated, mask=mask)
    # shape: [1]
    return torch.mean(log_density_data)

  def compute_all_losses(self, batch_dict, n_traj_samples=1, kl_coef=1.0):
    # Condition on subsampled points
    # Make predictions for all the points
    pred_y, info = self.get_reconstruction(
        batch_dict["tp_to_predict"],
        batch_dict["observed_data"],
        batch_dict["observed_tp"],
        mask=batch_dict["observed_mask"],
        n_traj_samples=n_traj_samples,
        mode=batch_dict["mode"],
    )

    # print('get_reconstruction done -- computing likelihood')
    fp_mu, fp_std, _ = info["first_point"]
    fp_distr = Normal(fp_mu, fp_std)

    assert torch.sum(fp_std < 0) == 0.0

    kldiv_z0 = kl_divergence(fp_distr, self.z0_prior)

    if torch.isnan(kldiv_z0).any():
      print(fp_mu)
      print(fp_std)
      raise ValueError("kldiv_z0 is Nan!")

    # Mean over number of latent dimensions
    # kldiv_z0 shape: [n_traj_samples, n_traj, n_latent_dims] if prior is a
    # mixture of gaussians (KL is estimated)
    # kldiv_z0 shape: [1, n_traj, n_latent_dims] if prior is a standard
    # gaussian (KL is computed exactly)
    # shape after: [n_traj_samples]
    kldiv_z0 = torch.mean(kldiv_z0, (1, 2))

    # Compute likelihood of all the points
    rec_likelihood = self.get_gaussian_likelihood(
        batch_dict["data_to_predict"],
        pred_y,
        mask=batch_dict["mask_predicted_data"],
    )

    mse = self.get_mse(
        batch_dict["data_to_predict"],
        pred_y,
        mask=batch_dict["mask_predicted_data"],
    )

    pois_log_likelihood = torch.Tensor([0.0]).to(batch_dict["data_to_predict"])
    if self.use_poisson_proc:
      pois_log_likelihood = compute_poisson_proc_likelihood(
          batch_dict["data_to_predict"],
          pred_y,
          info,
          mask=batch_dict["mask_predicted_data"],
      )
      # Take mean over n_traj
      pois_log_likelihood = torch.mean(pois_log_likelihood, 1)

    ################################
    # Compute CE loss for binary classification on Physionet
    ce_loss = torch.Tensor([0.0]).to(batch_dict["data_to_predict"])
    if (batch_dict["labels"] is not None) and self.use_binary_classif:

      if (batch_dict["labels"].size(-1) == 1) or (len(
          batch_dict["labels"].size()) == 1):
        ce_loss = compute_binary_CE_loss(info["label_predictions"],
                                         batch_dict["labels"])
      else:
        ce_loss = compute_multiclass_CE_loss(
            info["label_predictions"],
            batch_dict["labels"],
            mask=batch_dict["mask_predicted_data"],
        )

    # IWAE loss
    loss = -torch.logsumexp(rec_likelihood - kl_coef * kldiv_z0, 0)
    if torch.isnan(loss):
      loss = -torch.mean(rec_likelihood - kl_coef * kldiv_z0, 0)

    if self.use_poisson_proc:
      loss = loss - 0.1 * pois_log_likelihood

    if self.use_binary_classif:
      if self.train_classif_w_reconstr:
        loss = loss + ce_loss * 100
      else:
        loss = ce_loss

    results = {}
    results["loss"] = torch.mean(loss)
    results["likelihood"] = torch.mean(rec_likelihood).detach()
    results["mse"] = torch.mean(mse).detach()
    results["pois_likelihood"] = torch.mean(pois_log_likelihood).detach()
    results["ce_loss"] = torch.mean(ce_loss).detach()
    results["kl_first_p"] = torch.mean(kldiv_z0).detach()
    results["std_first_p"] = torch.mean(fp_std).detach()

    if batch_dict["labels"] is not None and self.use_binary_classif:
      results["label_predictions"] = info["label_predictions"].detach()

    return results


class LatentODE(VAE_Baseline):
  """
  Latent ODE model for variational autoencoders.

  Args:
    input_dim (int): Dimensionality of input data.
    latent_dim (int): Dimensionality of latent space.
    encoder_z0 (nn.Module): Encoder network for initial latent state.
    decoder (nn.Module): Decoder network.
    diffeq_solver (nn.Module): Differential equation solver.
    z0_prior (torch.distributions.Distribution): Prior distribution for initial
      latent state.
    device (torch.device): Device to run computations on.
    obsrv_std (float, optional): Observation standard deviation.
    use_binary_classif (bool, optional): Whether to use binary classification.
    use_poisson_proc (bool, optional): Whether to use Poisson process.
    linear_classifier (bool, optional): Whether to use a linear classifier.
    classif_per_tp (bool, optional): Whether to classify per time point.
    n_labels (int, optional): Number of labels for classification.
    train_classif_w_reconstr (bool, optional): Whether to train classifier with
      reconstruction.
  """

  def __init__(
      self,
      input_dim,
      latent_dim,
      encoder_z0,
      decoder,
      diffeq_solver,
      z0_prior,
      device,
      obsrv_std=None,
      use_binary_classif=False,
      use_poisson_proc=False,
      linear_classifier=False,
      classif_per_tp=False,
      n_labels=1,
      train_classif_w_reconstr=False,
  ):

    super(LatentODE, self).__init__(
        input_dim=input_dim,
        latent_dim=latent_dim,
        z0_prior=z0_prior,
        device=device,
        obsrv_std=obsrv_std,
        use_binary_classif=use_binary_classif,
        classif_per_tp=classif_per_tp,
        linear_classifier=linear_classifier,
        use_poisson_proc=use_poisson_proc,
        n_labels=n_labels,
        train_classif_w_reconstr=train_classif_w_reconstr,
    )

    self.encoder_z0 = encoder_z0
    self.diffeq_solver = diffeq_solver
    self.decoder = decoder
    self.use_poisson_proc = use_poisson_proc

  def get_reconstruction(
      self,
      time_steps_to_predict,
      truth,
      truth_time_steps,
      mask=None,
      n_traj_samples=1,
      run_backwards=True,
      mode=None, # pylint: disable=unused-argument
  ):

    assert isinstance(self.encoder_z0, Encoder_z0_ODE_RNN)

    truth_w_mask = truth
    if mask is not None:
      truth_w_mask = torch.cat((truth, mask), -1)

      first_point_mu, first_point_std = self.encoder_z0(
          truth_w_mask, truth_time_steps, run_backwards=run_backwards)

      means_z0 = first_point_mu.repeat(n_traj_samples, 1, 1)
      sigma_z0 = first_point_std.repeat(n_traj_samples, 1, 1)
      first_point_enc = sample_standard_gaussian(means_z0, sigma_z0)

    assert torch.sum(first_point_std < 0) == 0.0

    if self.use_poisson_proc:
      n_traj_samples, n_traj, _ = first_point_enc.size() # pylint: disable=possibly-used-before-assignment
      zeros = torch.zeros([n_traj_samples, n_traj, self.input_dim]).to(truth)
      first_point_enc_aug = torch.cat((first_point_enc, zeros), -1)
    else:
      first_point_enc_aug = first_point_enc

    assert not torch.isnan(time_steps_to_predict).any()
    assert not torch.isnan(first_point_enc).any()
    assert not torch.isnan(first_point_enc_aug).any()

    # sol_y shape [n_traj_samples, n_samples, n_timepoints, n_latents]
    initial_state = first_point_enc_aug.unsqueeze(-2)
    sol_y = self.diffeq_solver(initial_state,
                               time_steps_to_predict.unsqueeze(0))

    if self.use_poisson_proc:
      (
          sol_y,
          log_lambda_y,
          int_lambda,
          _,
      ) = self.diffeq_solver.ode_func.extract_poisson_rate(sol_y)

      assert torch.sum(int_lambda[:, :, 0, :]) == 0.0
      assert torch.sum(int_lambda[0, 0, -1, :] <= 0) == 0.0

    pred_x = self.decoder(sol_y)

    all_extra_info = {
        "first_point": (first_point_mu, first_point_std, first_point_enc),
        "latent_traj": sol_y.detach(),
    }

    if self.use_poisson_proc:
      all_extra_info["int_lambda"] = int_lambda[:, :, -1, :]
      all_extra_info["log_lambda_y"] = log_lambda_y

    if self.use_binary_classif:
      if self.classif_per_tp:
        all_extra_info["label_predictions"] = self.classifier(sol_y)
      else:
        all_extra_info["label_predictions"] = self.classifier(
            first_point_enc).squeeze(-1)

    return pred_x, all_extra_info

  def sample_traj_from_prior(self, time_steps_to_predict, n_traj_samples=1):
    starting_point_enc = self.z0_prior.sample(
        [n_traj_samples, 1, self.latent_dim]).squeeze(-1)

    starting_point_enc_aug = starting_point_enc
    if self.use_poisson_proc:
      n_traj_samples, n_traj, _ = starting_point_enc.size()
      zeros = torch.zeros(n_traj_samples, n_traj,
                          self.input_dim).to(self.device)
      starting_point_enc_aug = torch.cat((starting_point_enc, zeros), -1)

    sol_y = self.diffeq_solver.sample_traj_from_prior(starting_point_enc_aug,
                                                      time_steps_to_predict,
                                                      n_traj_samples=3)

    if self.use_poisson_proc:
      sol_y, _, _, _ = self.diffeq_solver.ode_func.extract_poisson_rate(sol_y)

    return self.decoder(sol_y)


def create_classifier(z0_dim, n_labels):
  return nn.Sequential(
      nn.Linear(z0_dim, 300),
      nn.ReLU(),
      nn.Linear(300, 300),
      nn.ReLU(),
      nn.Linear(300, n_labels),
  )
