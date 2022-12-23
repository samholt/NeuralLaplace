import torch
from torch import nn
from torchlaplace import laplace_reconstruct

class LaplaceRepresentationFunc(nn.Module):
    # SphereSurfaceModel : C^{b+k} -> C^{bxd} - In Riemann Sphere Co ords : b dim s reconstruction terms, k is latent encoding dimension, d is output dimension
    def __init__(self, s_dim, output_dim, latent_dim, hidden_units=64):
        super(LaplaceRepresentationFunc, self).__init__()
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

    def forward(self, i):
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

def test_laplace_reconstruct_original():
    s_recon_terms = 33
    output_dim = 1
    latent_dim = 2
    device = torch.device("cuda:" + str(0) if torch.cuda.is_available() else "cpu")
    laplace_rep_func = LaplaceRepresentationFunc(
        s_recon_terms, output_dim, latent_dim
    ).to(device)
    predictions = laplace_reconstruct(laplace_rep_func, torch.Tensor([[1.0,2.0], [1.0,2.0]]).to(device), torch.Tensor([1.0, 2.0, 3.0]).to(device))
    
def test_laplace_reconstruct_time_shared_for_across_all_samples_within_batch():
    s_recon_terms = 33
    output_dim = 1
    latent_dim = 2
    device = torch.device("cuda:" + str(0) if torch.cuda.is_available() else "cpu")
    laplace_rep_func = LaplaceRepresentationFunc(
        s_recon_terms, output_dim, latent_dim
    ).to(device)
    p = torch.rand((128,2)).to(device)
    t = torch.rand((1,100)).to(device)
    predictions = laplace_reconstruct(laplace_rep_func, p, t, recon_dim=1)
    assert predictions.shape[0] == 128
    assert predictions.shape[1] == 100

def test_laplace_reconstruct_time_shared_for_across_all_samples_within_batch_with_time_as_vector():
    s_recon_terms = 33
    output_dim = 1
    latent_dim = 2
    device = torch.device("cuda:" + str(0) if torch.cuda.is_available() else "cpu")
    laplace_rep_func = LaplaceRepresentationFunc(
        s_recon_terms, output_dim, latent_dim
    ).to(device)
    p = torch.rand((128,2)).to(device)
    t = torch.rand((100)).to(device)
    predictions = laplace_reconstruct(laplace_rep_func, p, t, recon_dim=1)
    assert predictions.shape[0] == 128
    assert predictions.shape[1] == 100

def test_laplace_reconstruct_unique_times_for_each_sample_within_batch():
    s_recon_terms = 33
    output_dim = 1
    latent_dim = 2
    device = torch.device("cuda:" + str(0) if torch.cuda.is_available() else "cpu")
    laplace_rep_func = LaplaceRepresentationFunc(
        s_recon_terms, output_dim, latent_dim
    ).to(device)
    p = torch.rand((128,2)).to(device)
    t = torch.rand((128,100)).to(device)
    predictions = laplace_reconstruct(laplace_rep_func, p, t, recon_dim=1)
    assert predictions.shape[0] == 128
    assert predictions.shape[1] == 100

if __name__ == "__main__":
    test_laplace_reconstruct_original()
    test_laplace_reconstruct_time_shared_for_across_all_samples_within_batch()
    test_laplace_reconstruct_time_shared_for_across_all_samples_within_batch_with_time_as_vector()
    test_laplace_reconstruct_unique_times_for_each_sample_within_batch()