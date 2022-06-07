###########################
# Neural Laplace: Learning diverse classes of differential equations in the Laplace domain
# Author: Samuel Holt
###########################
import logging

import torch
from torch import nn
from torchlaplace import laplace_reconstruct

logger = logging.getLogger()

class SphereSurfaceModel(nn.Module):
    # C^{b+k} -> C^{bxd} - In Riemann Sphere Co ords : b dim s reconstruction terms, k is latent encoding dimension, d is output dimension
    def __init__(
        self, s_dim, output_dim, latent_dim, hidden_units=64
    ):
        super(SphereSurfaceModel, self).__init__()
        self.s_dim = s_dim
        self.output_dim = output_dim
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
        self.nfe = 0

    def forward(self, i):
        self.nfe += 1
        out = self.linear_tanh_stack(i.view(-1, self.s_dim * 2 + self.latent_dim)).view(
            -1, 2 * self.output_dim, self.s_dim
        )
        theta = nn.Tanh()(out[:, : self.output_dim, :]) * torch.pi  # From - pi to + pi
        phi = (
            nn.Tanh()(out[:, self.output_dim :, :]) * self.phi_scale / 2.0
            - torch.pi / 2.0
            + self.phi_scale / 2.0
        )  # Form -pi / 2 to + pi / 2
        return theta, phi


class SurfaceModel(nn.Module):
    # C^{b+k} -> C^{bxd} - In Riemann Sphere Co ords : b dim s reconstruction terms, k is latent encoding dimension, d is output dimension
    def __init__(
        self, s_dim, output_dim, latent_dim, hidden_units=64
    ):
        super(SurfaceModel, self).__init__()
        self.s_dim = s_dim
        self.output_dim = output_dim
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

        self.nfe = 0

    def forward(self, i):
        self.nfe += 1
        out = self.linear_tanh_stack(i.view(-1, self.s_dim * 2 + self.latent_dim)).view(
            -1, 2 * self.output_dim, self.s_dim
        )
        real = out[:, : self.output_dim, :]
        imag = out[:, self.output_dim :, :]
        return real, imag

class ReverseGRUEncoder(nn.Module):
    # Encodes observed trajectory into latent vector
    def __init__(self, dimension_in, latent_dim, hidden_units, encode_obs_time=True):
        super(ReverseGRUEncoder, self).__init__()
        self.encode_obs_time = encode_obs_time
        if self.encode_obs_time:
            dimension_in += 1
        self.gru = nn.GRU(
            dimension_in, hidden_units, 2, batch_first=True)
        self.linear_out = nn.Linear(hidden_units, latent_dim)
        nn.init.xavier_uniform_(self.linear_out.weight)

    def forward(self, observed_data, observed_tp):
        trajs_to_encode = observed_data # (batch_size, t_observed_dim, observed_dim)
        if self.encode_obs_time:
            trajs_to_encode = torch.cat((observed_data, observed_tp.view(1,-1,1).repeat(observed_data.shape[0],1,1)), dim=2)
        reversed_trajs_to_encode = torch.flip(trajs_to_encode, (1,))
        out, _ = self.gru(reversed_trajs_to_encode)
        return self.linear_out(out[:, -1, :])

class NeuralLaplace(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        latent_dim=2,
        hidden_units=64,
        s_recon_terms=33,
        use_sphere_projection=True,
        encode_obs_time=True,
        ilt_algorithm="fourier",
    ):
        super(NeuralLaplace, self).__init__()
        self.encoder = ReverseGRUEncoder(input_dim, latent_dim, hidden_units // 2, encode_obs_time=encode_obs_time)
        self.use_sphere_projection = use_sphere_projection
        self.output_dim = output_dim
        self.ilt_algorithm = ilt_algorithm
        if use_sphere_projection:
            self.laplace_rep_func = SphereSurfaceModel(
                s_recon_terms, output_dim, latent_dim
            )
        else:
            self.laplace_rep_func = SurfaceModel(
                s_recon_terms, output_dim, latent_dim
            )

    def forward(self, observed_data, observed_tp, tp_to_predict):
        # trajs_to_encode : (N, T, D) tensor containing the observed values.
        # tp_to_predict: Is the time to predict the values at.
        p = self.encoder(observed_data, observed_tp)
        return laplace_reconstruct(self.laplace_rep_func, p, tp_to_predict, recon_dim=self.output_dim, use_sphere_projection=self.use_sphere_projection, ilt_algorithm=self.ilt_algorithm)

class GeneralNeuralLaplace(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        latent_dim=2,
        hidden_units=64,
        s_recon_terms=33,
        use_sphere_projection=True,
        encode_obs_time=True,
        ilt_algorithm="fourier",

    ):
        super(GeneralNeuralLaplace, self).__init__()

        self.model = NeuralLaplace(
            input_dim,
            output_dim,
            latent_dim=latent_dim,
            hidden_units=hidden_units,
            s_recon_terms=s_recon_terms,
            use_sphere_projection=use_sphere_projection,
            encode_obs_time=encode_obs_time,
            ilt_algorithm=ilt_algorithm,
        )
        self.loss_fn = torch.nn.MSELoss()

    def _get_loss(self, dl):
        cum_loss = 0
        cum_batches = 0
        for batch in dl:
            preds = self.model(batch["observed_data"], batch["observed_tp"], batch["tp_to_predict"])
            cum_loss += self.loss_fn(
                torch.flatten(preds), torch.flatten(batch["data_to_predict"])
            )
            cum_batches += 1
        mse = cum_loss / cum_batches
        return mse

    def training_step(self, batch):
        preds = self.model(batch["observed_data"], batch["observed_tp"], batch["tp_to_predict"])
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
                self.model(batch["observed_data"], batch["observed_tp"], batch["tp_to_predict"])
            )
        return torch.cat(predictions, 0)

    def encode(self, dl):
        encodings = []
        for batch in dl:
            encodings.append(self.model.encode(batch["observed_data"], batch["observed_tp"]))
        return torch.cat(encodings, 0)

    def _get_and_reset_nfes(self):
        """Returns and resets the number of function evaluations for model."""
        iteration_nfes = self.model.laplace_rep_func.nfe
        self.model.laplace_rep_func.nfe = 0
        return iteration_nfes
