import logging

import torch
from torch import nn
from torchdiffeq import odeint_adjoint as odeint

logger = logging.getLogger()


device = "cuda" if torch.cuda.is_available() else "cpu"


class OdeFunc(nn.Module):
    def __init__(self, obs_dim=2, nhidden=50, time_dependent=True):
        super(OdeFunc, self).__init__()
        self.time_dependent = time_dependent
        self.sig = nn.Tanh()
        if time_dependent:
            self.fc1 = nn.Linear(obs_dim + 1, nhidden)
        else:
            self.fc1 = nn.Linear(obs_dim, nhidden)
        self.fc1_5 = nn.Linear(nhidden, nhidden)
        self.fc2 = nn.Linear(nhidden, obs_dim)
        self.nfe = 0

    def forward(self, t, z):
        self.nfe += 1
        if self.time_dependent:
            # Shape (batch_size, 1)
            t_vec = torch.ones(z.shape[0], 1).to(device) * t
            # Shape (batch_size, data_dim + 1)
            t_and_x = torch.cat([t_vec, z], 1)
            # Shape (batch_size, hidden_dim)
            out = self.fc1(t_and_x)
        else:
            out = self.fc1(z)
        out = self.sig(out)
        out = self.fc1_5(out)
        out = self.sig(out)
        out = self.fc2(out)
        return out


class NODE(nn.Module):
    def __init__(self, obs_dim=2, nhidden=50, method="euler", augment_dim=0, extrap=0):
        super(NODE, self).__init__()
        self.ode_func = OdeFunc(obs_dim + augment_dim, nhidden)
        self.method = method
        self.augment_dim = augment_dim
        self.extrap = extrap

    def encode(self, trajectories):
        if self.extrap:
            x0 = trajectories[:, -1, :]
        else:
            x0 = trajectories[:, 0, :]
        return x0

    def forward(self, trajectories, ti):
        # Trajectories : (N, T, D) tensor containing the observed values.
        t = torch.flatten(ti)

        if self.extrap:
            x0 = trajectories[:, -1, :]
        else:
            x0 = trajectories[:, 0, :]

        if self.augment_dim > 0:
            # Add augmentation
            aug = torch.zeros(x0.shape[0], self.augment_dim).to(device)
            # Shape (batch_size, data_dim + augment_dim)
            x_aug = torch.cat([x0, aug], 1)
        else:
            x_aug = x0

        features = odeint(
            self.ode_func,
            x_aug,
            t,
            method=self.method,
            adjoint_options={"norm": "seminorm"},
        )

        features = torch.transpose(features, 0, 1)
        return features[:, :, : features.shape[2] - self.augment_dim]


class LatentNODE(nn.Module):
    def __init__(
        self,
        obs_dim=2,
        nhidden=50,
        latent_dim=2,
        method="euler",
        augment_dim=0,
        extrap=0,
    ):
        super(LatentNODE, self).__init__()
        self.ode_func = OdeFunc(obs_dim + augment_dim, nhidden)
        self.method = method
        self.encoder = GRUEncoder(latent_dim, nhidden)
        self.augment_dim = augment_dim
        self.extrap = extrap

    def forward(self, trajectories, ti):
        # Trajectories : (N, T, D) tensor containing the observed values.
        t = torch.flatten(ti)
        x0 = self.encoder(trajectories)

        if self.augment_dim > 0:
            # Add augmentation
            aug = torch.zeros(x0.shape[0], self.augment_dim).to(device)
            # Shape (batch_size, data_dim + augment_dim)
            x_aug = torch.cat([x0, aug], 1)
        else:
            x_aug = x0

        features = odeint(
            self.ode_func,
            x_aug,
            t,
            method=self.method,
            adjoint_options={"norm": "seminorm"},
        )
        features = torch.transpose(features, 0, 1)
        return features[:, :, features.shape[2] - self.augment_dim :]


class GRUEncoder(nn.Module):
    def __init__(self, dimension, hidden_units):
        super(GRUEncoder, self).__init__()
        self.gru = nn.GRU(dimension, hidden_units, 2, batch_first=True)
        self.linear_out = nn.Linear(hidden_units, dimension)
        nn.init.xavier_uniform_(self.linear_out.weight)

    def forward(self, i):
        out, _ = self.gru(i)
        return self.linear_out(out[:, -1, :])


class GeneralNODE(nn.Module):
    def __init__(
        self,
        obs_dim=2,
        nhidden=50,
        method="euler",
        latent_dim=2,
        augment_dim=0,
        extrap=0,
    ):
        super(GeneralNODE, self).__init__()
        self.model = NODE(obs_dim, nhidden, method, augment_dim, extrap)
        self.loss_fn = torch.nn.MSELoss()

    def _get_loss(self, dl):
        cum_loss = 0
        cum_batches = 0
        for batch in dl:
            preds = self.model(batch["observed_data"], batch["tp_to_predict"])
            cum_loss += self.loss_fn(
                torch.flatten(preds), torch.flatten(batch["data_to_predict"])
            )
            cum_batches += 1
        mse = cum_loss / cum_batches
        return mse

    def training_step(self, batch):
        preds = self.model(batch["observed_data"], batch["tp_to_predict"])
        return self.loss_fn(
            torch.flatten(preds), torch.flatten(batch["data_to_predict"])
        )

    def validation_step(self, dlval):
        mse = self._get_loss(dlval)
        return mse, mse

    def test_step(self, dltest):
        mse = self._get_loss(dltest)
        return mse, mse

    def predict(self, dl):
        predictions = []
        for batch in dl:
            predictions.append(
                self.model(batch["observed_data"], batch["tp_to_predict"])
            )
        return torch.cat(predictions, 0)

    def encode(self, dl):
        encodings = []
        for batch in dl:
            encodings.append(self.model.encode(batch["observed_data"]))
        return torch.cat(encodings, 0)

    def _get_and_reset_nfes(self):
        """Returns and resets the number of function evaluations for model."""
        iteration_nfes = self.model.ode_func.nfe
        self.model.ode_func.nfe = 0
        return iteration_nfes
