# Ref: Neural Flows
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Independent, kl_divergence
from torch.distributions.normal import Normal

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = Path("./data")
LOCAL = False


def basic_collate_fn(
    batch,
    time_steps,
    extrap=False,
    data="",
    device=DEVICE,
    data_type="train",
    observe_step=1,
    predict_step=1,
):
    batch = torch.stack(batch)
    data_dict = {
        "data": batch,
        "time_steps": time_steps.unsqueeze(0),
    }
    data_dict = split_and_subsample_batch(
        data_dict,
        extrap=extrap,
        data=data,
        data_type=data_type,
        observe_step=observe_step,
        predict_step=predict_step,
    )
    return data_dict


def variable_time_collate_fn(
    batch,
    extrap=0,
    data="",
    device=DEVICE,
    data_type="train",
    data_min=None,
    data_max=None,
):
    """
    Expects a batch of time series data in the form of (record_id, tt, vals, mask, labels) where
            - record_id is a patient id
            - tt is a 1-dimensional tensor containing T time values of observations.
            - vals is a (T, D) tensor containing observed values for D variables.
            - mask is a (T, D) tensor containing 1 where values were observed and 0 otherwise.
            - labels is a list of labels for the current patient, if labels are available. Otherwise None.
    Returns:
            combined_tt: The union of all time observations.
            combined_vals: (M, T, D) tensor containing the observed values.
            combined_mask: (M, T, D) tensor containing 1 where values were observed and 0 otherwise.
    """
    D = batch[0][2].shape[1]

    lens = [len(x[1]) for x in batch]
    max_len = max(lens)

    combined_tt = torch.stack(
        [torch.cat([x[1], torch.zeros(max_len - l)], 0) for x, l in zip(batch, lens)], 0
    ).double()
    combined_vals = torch.stack(
        [
            torch.cat([x[2], torch.zeros(max_len - l, D)], 0)
            for x, l in zip(batch, lens)
        ],
        0,
    ).double()
    combined_mask = torch.stack(
        [
            torch.cat([x[3], torch.zeros(max_len - l, D)], 0)
            for x, l in zip(batch, lens)
        ],
        0,
    ).double()
    combined_labels = (
        torch.stack(
            [torch.tensor(float("nan")) if x[4] is None else x[4] for x in batch], 0
        )
        .to(device)
        .double()
    )
    combined_labels = combined_labels.unsqueeze(-1)

    # Normalize data an time
    combined_vals, _, _ = normalize_masked_data(
        combined_vals, combined_mask, data_min, data_max
    )
    combined_tt = combined_tt / 48.0

    data_dict = {
        "data": combined_vals,  # (batch, sequence length, dim=41)
        "time_steps": combined_tt,  # (batch, sequence length)
        "mask": combined_mask,  # (batch, sequence length, dim=41)
        "labels": combined_labels,  # (batch, 1)
    }

    data_dict = split_and_subsample_batch(data_dict, extrap=extrap, data_type=data_type)
    return data_dict


def split_and_subsample_batch(
    data_dict, extrap=False, data="", data_type="train", observe_step=1, predict_step=1
):
    if data_type == "train":
        # Training set
        if extrap:
            processed_dict = split_data_extrap(
                data_dict,
                dataset=data,
                observe_step=observe_step,
                predict_step=predict_step,
            )
        else:
            processed_dict = split_data_interp(data_dict)
    else:
        # Test set
        if extrap:
            processed_dict = split_data_extrap(
                data_dict,
                dataset=data,
                observe_step=observe_step,
                predict_step=predict_step,
            )
        else:
            processed_dict = split_data_interp(data_dict)

    # add mask
    processed_dict = add_mask(processed_dict)
    return processed_dict


def split_last_dim(data):
    last_dim = data.size()[-1]
    last_dim = last_dim // 2

    if len(data.size()) == 3:
        res = data[:, :, :last_dim], data[:, :, last_dim:]

    if len(data.size()) == 2:
        res = data[:, :last_dim], data[:, last_dim:]
    return res


def init_network_weights(net, std=0.1):
    for m in net.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0, std=std)
            nn.init.constant_(m.bias, val=0)


def get_device(tensor):
    device = torch.device("cpu")
    if tensor.is_cuda:
        device = tensor.get_device()
    return device


def sample_standard_gaussian(mu, sigma):
    device = get_device(mu)

    d = torch.distributions.normal.Normal(
        torch.Tensor([0.0]).to(device), torch.Tensor([1.0]).to(device)
    )
    r = d.sample(mu.size()).squeeze(-1)
    return r * sigma.float() + mu.float()


def split_train_val_test(data, train_frac=0.6, val_frac=0.2):
    n_samples = len(data)
    data_train = data[: int(n_samples * train_frac)]
    data_val = data[
        int(n_samples * train_frac) : int(n_samples * (train_frac + val_frac))
    ]
    data_test = data[int(n_samples * (train_frac + val_frac)) :]
    return data_train, data_val, data_test


def linspace_vector(start, end, n_points):
    # start is either one value or a vector
    size = np.prod(start.size())

    if start.size() != end.size():
        raise ValueError("Invalid start end size")
    if size == 1:
        # start and end are 1d-tensors
        res = torch.linspace(start, end, n_points)
    else:
        # start and end are vectors
        res = torch.Tensor()
        for i in range(0, start.size(0)):
            res = torch.cat((res, torch.linspace(start[i], end[i], n_points)), 0)
        res = torch.t(res.reshape(start.size(0), n_points))
    return res


def reverse(tensor):
    idx = [i for i in range(tensor.size(0) - 1, -1, -1)]
    return tensor[idx]


def create_net(n_inputs, n_outputs, n_layers=1, n_units=100, nonlinear=nn.Tanh):
    layers = [nn.Linear(n_inputs, n_units)]
    for i in range(n_layers):
        layers.append(nonlinear())
        layers.append(nn.Linear(n_units, n_units))

    layers.append(nonlinear())
    layers.append(nn.Linear(n_units, n_outputs))
    return nn.Sequential(*layers)


def get_dict_template():
    return {
        "observed_data": None,
        "observed_tp": None,
        "data_to_predict": None,
        "tp_to_predict": None,
        "observed_mask": None,
        "mask_predicted_data": None,
        "labels": None,
    }


def normalize_data(data):
    reshaped = data.reshape(-1, data.size(-1))

    att_min = torch.min(reshaped, 0)[0]
    att_max = torch.max(reshaped, 0)[0]

    # we don't want to divide by zero
    att_max[att_max == 0.0] = 1.0

    if (att_max != 0.0).all():
        data_norm = (data - att_min) / att_max
    else:
        raise Exception("Zero!")

    if torch.isnan(data_norm).any():
        raise Exception("nans!")

    return data_norm, att_min, att_max


def normalize_masked_data(data, mask, att_min, att_max):
    # we don't want to divide by zero
    att_max[att_max == 0.0] = 1.0

    if (att_max != 0.0).all():
        data_norm = (data - att_min) / att_max
    else:
        raise Exception("Zero!")

    if torch.isnan(data_norm).any():
        raise Exception("nans!")

    # set masked out elements back to zero
    data_norm[mask == 0] = 0

    return data_norm, att_min, att_max


def shift_outputs(outputs, first_datapoint=None):
    outputs = outputs[:, :, :-1, :]

    if first_datapoint is not None:
        n_traj, n_dims = first_datapoint.size()
        first_datapoint = first_datapoint.reshape(1, n_traj, 1, n_dims)
        outputs = torch.cat((first_datapoint, outputs), 2)
    return outputs


def split_data_extrap(data_dict, dataset="", observe_step=1, predict_step=1):
    n_observed_tp = data_dict["data"].size(1) // 2
    if dataset == "hopper":
        n_observed_tp = data_dict["data"].size(1) // 3

    split_dict = {
        "observed_data": data_dict["data"][:, :n_observed_tp, :][
            :, ::observe_step, :
        ].clone(),
        "observed_tp": data_dict["time_steps"][:, :n_observed_tp][
            :, ::observe_step
        ].clone(),
        "data_to_predict": data_dict["data"][:, n_observed_tp:, :][
            :, ::predict_step, :
        ].clone(),
        "tp_to_predict": data_dict["time_steps"][:, n_observed_tp:][
            :, ::predict_step
        ].clone(),
    }

    split_dict["observed_mask"] = None
    split_dict["mask_predicted_data"] = None
    split_dict["labels"] = None

    if ("mask" in data_dict) and (data_dict["mask"] is not None):
        split_dict["observed_mask"] = data_dict["mask"][:, :n_observed_tp].clone()
        split_dict["mask_predicted_data"] = data_dict["mask"][:, n_observed_tp:].clone()

    if ("labels" in data_dict) and (data_dict["labels"] is not None):
        split_dict["labels"] = data_dict["labels"].clone()

    split_dict["mode"] = "extrap"
    return split_dict


def split_data_interp(data_dict):
    split_dict = {
        "observed_data": data_dict["data"].clone(),
        "observed_tp": data_dict["time_steps"].clone(),
        "data_to_predict": data_dict["data"].clone(),
        "tp_to_predict": data_dict["time_steps"].clone(),
    }

    split_dict["observed_mask"] = None
    split_dict["mask_predicted_data"] = None
    split_dict["labels"] = None

    if "mask" in data_dict and data_dict["mask"] is not None:
        split_dict["observed_mask"] = data_dict["mask"].clone()
        split_dict["mask_predicted_data"] = data_dict["mask"].clone()

    if ("labels" in data_dict) and (data_dict["labels"] is not None):
        split_dict["labels"] = data_dict["labels"].clone()

    split_dict["mode"] = "interp"
    return split_dict


def add_mask(data_dict):
    data = data_dict["observed_data"]
    mask = data_dict["observed_mask"]

    if mask is None:
        mask = torch.ones_like(data).to(get_device(data))

    data_dict["observed_mask"] = mask
    return data_dict


def compute_loss_all_batches(
    model,
    dl,
    device,
    classify=0,
    data="",
    n_traj_samples=3,
    kl_coef=1.0,
    max_samples_for_eval=None,
):
    total = {}
    total["loss"] = 0
    total["likelihood"] = 0
    total["mse"] = 0
    total["acc"] = 0
    total["kl_first_p"] = 0
    total["std_first_p"] = 0
    total["pois_likelihood"] = 0
    total["ce_loss"] = 0

    n_test_batches = 0

    classif_predictions = torch.Tensor([]).to(device)
    all_test_labels = torch.Tensor([]).to(device)

    for batch_dict in dl:
        results = model.compute_all_losses(
            batch_dict, n_traj_samples=n_traj_samples, kl_coef=kl_coef
        )

        if classify and data != "hopper":
            n_labels = model.n_labels  # batch_dict['labels'].size(-1)
            n_traj_samples = results["label_predictions"].size(0)

            classif_predictions = torch.cat(
                (
                    classif_predictions,
                    results["label_predictions"].reshape(n_traj_samples, -1, n_labels),
                ),
                1,
            )
            all_test_labels = torch.cat(
                (all_test_labels, batch_dict["labels"].reshape(-1, n_labels)), 0
            )

        for key in total.keys():
            if key in results:
                var = results[key]
                if isinstance(var, torch.Tensor):
                    var = var.detach()
                total[key] += var

        n_test_batches += 1

    if n_test_batches > 0:
        for key, _ in total.items():
            total[key] = total[key] / n_test_batches

    if classify:
        if data == "physionet":
            all_test_labels = all_test_labels.repeat(n_traj_samples, 1, 1)

            idx_not_nan = ~torch.isnan(all_test_labels)
            classif_predictions = classif_predictions[idx_not_nan]
            all_test_labels = all_test_labels[idx_not_nan]

            total["acc"] = 0.0  # AUC score
            if torch.sum(all_test_labels) != 0.0:
                print(f"Number of labeled examples: {len(all_test_labels.reshape(-1))}")
                print(
                    "Number of examples with mortality 1: {}".format(
                        torch.sum(all_test_labels == 1.0)
                    )
                )

                # Cannot compute AUC with only 1 class
                import sklearn as sk

                total["acc"] = sk.metrics.roc_auc_score(
                    all_test_labels.cpu().numpy().reshape(-1),
                    classif_predictions.cpu().numpy().reshape(-1),
                )
            else:
                print(
                    "Warning: Couldn't compute AUC -- all examples are from the same class"
                )

        if data == "activity":
            all_test_labels = all_test_labels.repeat(n_traj_samples, 1, 1)

            labeled_tp = torch.sum(all_test_labels, -1, keepdim=True) > 0.0
            labeled_tp = labeled_tp.repeat_interleave(all_test_labels.shape[-1], -1)

            all_test_labels = all_test_labels[labeled_tp].view(
                -1, all_test_labels.shape[-1]
            )
            classif_predictions = classif_predictions[labeled_tp].view(
                -1, all_test_labels.shape[-1]
            )

            # classif_predictions and all_test_labels are in on-hot-encoding -- convert to class ids
            _, pred_class_id = torch.max(classif_predictions, -1)
            _, class_labels = torch.max(all_test_labels, -1)

            total["acc"] = torch.sum(class_labels == pred_class_id).item() / sum(
                class_labels.shape
            )

    return total


def check_mask(data, mask):
    # check that 'mask' argument indeed contains a mask for data
    n_zeros = torch.sum(mask == 0.0).cpu().numpy()
    n_ones = torch.sum(mask == 1.0).cpu().numpy()

    # mask should contain only zeros and ones
    if (n_zeros + n_ones) != np.prod(list(mask.size())):
        raise ValueError("Invalid one zero mask")

    # all masked out elements should be zeros
    if torch.sum(data[mask == 0.0] != 0.0) != 0:
        raise ValueError("Invalid mask sum")


def create_classifier(z0_dim, n_labels):
    return nn.Sequential(
        nn.Linear(z0_dim, 300),
        nn.ReLU(),
        nn.Linear(300, 300),
        nn.ReLU(),
        nn.Linear(300, n_labels),
    )


class VAE_Baseline(nn.Module):
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
        n_traj, n_tp, n_dim = truth.size()

        # Compute likelihood of the data under the predictions
        truth_repeated = truth.repeat(pred_y.size(0), 1, 1, 1)

        if mask is not None:
            mask = mask.repeat(pred_y.size(0), 1, 1, 1)
        log_density_data = masked_gaussian_log_density(
            pred_y, truth_repeated, obsrv_std=self.obsrv_std, mask=mask
        )
        log_density_data = log_density_data.permute(1, 0)
        log_density = torch.mean(log_density_data, 1)

        # shape: [n_traj_samples]
        return log_density

    def get_mse(self, truth, pred_y, mask=None):
        # pred_y shape [n_traj_samples, n_traj, n_tp, n_dim]
        # truth shape  [n_traj, n_tp, n_dim]
        n_traj, n_tp, n_dim = truth.size()

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
        fp_mu, fp_std, fp_enc = info["first_point"]
        fp_distr = Normal(fp_mu, fp_std)

        if torch.sum(fp_std < 0) != 0.0:
            raise ValueError("Invalid fp_std")

        kldiv_z0 = kl_divergence(fp_distr, self.z0_prior)

        if torch.isnan(kldiv_z0).any():
            print(fp_mu)
            print(fp_std)
            raise Exception("kldiv_z0 is Nan!")

        # Mean over number of latent dimensions
        # kldiv_z0 shape: [n_traj_samples, n_traj, n_latent_dims] if prior is a mixture of gaussians (KL is estimated)
        # kldiv_z0 shape: [1, n_traj, n_latent_dims] if prior is a standard gaussian (KL is computed exactly)
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

            if (batch_dict["labels"].size(-1) == 1) or (
                len(batch_dict["labels"].size()) == 1
            ):
                ce_loss = compute_binary_CE_loss(
                    info["label_predictions"], batch_dict["labels"]
                )
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


def get_mask(x):
    x = x.unsqueeze(0)
    n_data_dims = x.size(-1) // 2
    mask = x[:, :, n_data_dims:]
    check_mask(x[:, :, :n_data_dims], mask)
    mask = (torch.sum(mask, -1, keepdim=True) > 0).float()
    if torch.isnan(mask).any():
        raise ValueError("Mask contains NaNs")
    return mask.squeeze(0)


class Encoder_z0_ODE_RNN(nn.Module):
    def __init__(
        self,
        latent_dim,
        input_dim,
        z0_diffeq_solver=None,
        z0_dim=None,
        n_gru_units=100,
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

    def forward(self, data, time_steps, run_backwards=True, save_info=False):
        if torch.isnan(data).any():
            raise ValueError("Mask should not contain NaNs")
        if torch.isnan(time_steps).any():
            raise ValueError("Time steps should not contain NaNs")

        n_traj, n_tp, n_dims = data.size()
        latent = self.run_odernn(data, time_steps, run_backwards)

        latent = latent.reshape(1, n_traj, self.latent_dim)

        mean_z0, std_z0 = self.transform_z0(latent).chunk(2, dim=-1)
        std_z0 = F.softplus(std_z0)

        return mean_z0, std_z0

    def run_odernn(self, data, time_steps, run_backwards=True):
        batch_size, n_tp, n_dims = data.size()
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
    def __init__(self, latent_dim, input_dim):
        super().__init__()
        decoder = nn.Sequential(
            nn.Linear(latent_dim, input_dim),
        )
        init_network_weights(decoder)
        self.decoder = decoder

    def forward(self, data):
        return self.decoder(data)


def gaussian_log_likelihood(mu_2d, data_2d, obsrv_std, indices=None):
    n_data_points = mu_2d.size()[-1]

    if n_data_points > 0:
        gaussian = Independent(
            Normal(loc=mu_2d, scale=obsrv_std.repeat(n_data_points)), 1
        )
        log_prob = gaussian.log_prob(data_2d)
        log_prob = log_prob / n_data_points
    else:
        log_prob = torch.zeros([1]).to(data_2d).squeeze()
    return log_prob


def poisson_log_likelihood(masked_log_lambdas, masked_data, indices, int_lambdas):
    # masked_log_lambdas and masked_data
    n_data_points = masked_data.size()[-1]

    if n_data_points > 0:
        log_prob = torch.sum(masked_log_lambdas) - int_lambdas[indices]
        # log_prob = log_prob / n_data_points
    else:
        log_prob = torch.zeros([1]).to(masked_data).squeeze()
    return log_prob


def compute_binary_CE_loss(label_predictions, mortality_label):
    # print('Computing binary classification loss: compute_CE_loss')

    mortality_label = mortality_label.reshape(-1)

    if len(label_predictions.size()) == 1:
        label_predictions = label_predictions.unsqueeze(0)

    n_traj_samples = label_predictions.size(0)
    label_predictions = label_predictions.reshape(n_traj_samples, -1)

    idx_not_nan = ~torch.isnan(mortality_label)
    if len(idx_not_nan) == 0.0:
        print("All are labels are NaNs!")
        ce_loss = torch.Tensor(0.0).to(mortality_label)

    label_predictions = label_predictions[:, idx_not_nan]
    mortality_label = mortality_label[idx_not_nan]

    if torch.sum(mortality_label == 0.0) == 0 or torch.sum(mortality_label == 1.0) == 0:
        print(
            "Warning: all examples in a batch belong to the same class -- please increase the batch size."
        )

    if torch.isnan(label_predictions).any():
        raise ValueError("label_predictions contain NaNs")
    if torch.isnan(mortality_label).any():
        raise ValueError("mortality_label contains NaNs")

    # For each trajectory, we get n_traj_samples samples from z0 -- compute loss on all of them
    mortality_label = mortality_label.repeat(n_traj_samples, 1)
    ce_loss = nn.BCEWithLogitsLoss()(label_predictions, mortality_label)

    # divide by number of patients in a batch
    ce_loss = ce_loss / n_traj_samples
    return ce_loss


def compute_multiclass_CE_loss(label_predictions, true_label, mask):
    # print('Computing multi-class classification loss: compute_multiclass_CE_loss')

    if len(label_predictions.size()) == 3:
        label_predictions = label_predictions.unsqueeze(0)

    n_traj_samples, n_traj, n_tp, n_dims = label_predictions.size()

    # For each trajectory, we get n_traj_samples samples from z0 -- compute loss on all of them
    true_label = true_label.repeat(n_traj_samples, 1, 1)

    label_predictions = label_predictions.reshape(
        n_traj_samples * n_traj * n_tp, n_dims
    )
    true_label = true_label.reshape(n_traj_samples * n_traj * n_tp, n_dims)

    # choose time points with at least one measurement
    mask = torch.sum(mask, -1) > 0

    # repeat the mask for each label to mark that the label for this time point is present
    pred_mask = mask.repeat(n_dims, 1, 1).permute(1, 2, 0)

    label_mask = mask
    pred_mask = pred_mask.repeat(n_traj_samples, 1, 1, 1)
    label_mask = label_mask.repeat(n_traj_samples, 1, 1, 1)

    pred_mask = pred_mask.reshape(n_traj_samples * n_traj * n_tp, n_dims)
    label_mask = label_mask.reshape(n_traj_samples * n_traj * n_tp, 1)

    if (label_predictions.size(-1) > 1) and (true_label.size(-1) > 1):
        if label_predictions.size(-1) != true_label.size(-1):
            raise RuntimeError("Invalid label_predictions shape")
        # targets are in one-hot encoding -- convert to indices
        _, true_label = true_label.max(-1)

    res = []
    for i in range(true_label.size(0)):
        pred_masked = torch.masked_select(label_predictions[i], pred_mask[i].bool())
        labels = torch.masked_select(true_label[i], label_mask[i].bool())

        pred_masked = pred_masked.reshape(-1, n_dims)

        if len(labels) == 0:
            continue

        ce_loss = nn.CrossEntropyLoss()(pred_masked, labels.long())
        res.append(ce_loss)

    ce_loss = torch.stack(res, 0).to(label_predictions)
    ce_loss = torch.mean(ce_loss)
    # # divide by number of patients in a batch
    # ce_loss = ce_loss / n_traj_samples
    return ce_loss


def compute_masked_likelihood(mu, data, mask, likelihood_func):
    # Compute the likelihood per patient and per attribute so that we don't priorize patients with more measurements
    n_traj_samples, n_traj, n_timepoints, n_dims = data.size()

    res = []
    for i in range(n_traj_samples):
        for k in range(n_traj):
            for j in range(n_dims):
                data_masked = torch.masked_select(
                    data[i, k, :, j], mask[i, k, :, j].bool()
                )

                mu_masked = torch.masked_select(mu[i, k, :, j], mask[i, k, :, j].bool())
                log_prob = likelihood_func(mu_masked, data_masked, indices=(i, k, j))
                res.append(log_prob)
    # shape: [n_traj*n_traj_samples, 1]

    res = torch.stack(res, 0).to(data)
    res = res.reshape((n_traj_samples, n_traj, n_dims))
    # Take mean over the number of dimensions
    res = torch.mean(res, -1)  # !!!!!!!!!!! changed from sum to mean
    res = res.transpose(0, 1)
    return res


def masked_gaussian_log_density(mu, data, obsrv_std, mask=None):
    # these cases are for plotting through plot_estim_density
    if len(mu.size()) == 3:
        # add additional dimension for gp samples
        mu = mu.unsqueeze(0)

    if len(data.size()) == 2:
        # add additional dimension for gp samples and time step
        data = data.unsqueeze(0).unsqueeze(2)
    elif len(data.size()) == 3:
        # add additional dimension for gp samples
        data = data.unsqueeze(0)

    n_traj_samples, n_traj, n_timepoints, n_dims = mu.size()

    if data.size()[-1] != n_dims:
        raise ValueError("Invalid data size")
    # Shape after permutation: [n_traj, n_traj_samples, n_timepoints, n_dims]
    if mask is None:
        mu_flat = mu.reshape(n_traj_samples * n_traj, n_timepoints * n_dims)
        n_traj_samples, n_traj, n_timepoints, n_dims = data.size()
        data_flat = data.reshape(n_traj_samples * n_traj, n_timepoints * n_dims)

        res = gaussian_log_likelihood(mu_flat, data_flat, obsrv_std)
        res = res.reshape(n_traj_samples, n_traj).transpose(0, 1)
    else:
        # Compute the likelihood per patient so that we don't priorize patients with more measurements
        def func(mu, data, indices):
            return gaussian_log_likelihood(
                mu, data, obsrv_std=obsrv_std, indices=indices
            )

        res = compute_masked_likelihood(mu, data, mask, func)
    return res


def mse(mu, data, indices=None):
    n_data_points = mu.size()[-1]

    if n_data_points > 0:
        mse = nn.MSELoss()(mu, data)
    else:
        mse = torch.zeros([1]).to(data).squeeze()
    return mse


def compute_mse(mu, data, mask=None):
    # these cases are for plotting through plot_estim_density
    if len(mu.size()) == 3:
        # add additional dimension for gp samples
        mu = mu.unsqueeze(0)

    if len(data.size()) == 2:
        # add additional dimension for gp samples and time step
        data = data.unsqueeze(0).unsqueeze(2)
    elif len(data.size()) == 3:
        # add additional dimension for gp samples
        data = data.unsqueeze(0)

    n_traj_samples, n_traj, n_timepoints, n_dims = mu.size()
    if data.size()[-1] != n_dims:
        raise ValueError("Invalid data size")

    # Shape after permutation: [n_traj, n_traj_samples, n_timepoints, n_dims]
    if mask is None:
        mu_flat = mu.reshape(n_traj_samples * n_traj, n_timepoints * n_dims)
        n_traj_samples, n_traj, n_timepoints, n_dims = data.size()
        data_flat = data.reshape(n_traj_samples * n_traj, n_timepoints * n_dims)
        res = mse(mu_flat, data_flat)
    else:
        # Compute the likelihood per patient so that we don't priorize patients with more measurements
        res = compute_masked_likelihood(mu, data, mask, mse)
    return res


def compute_poisson_proc_likelihood(truth, pred_y, info, mask=None):
    # Compute Poisson likelihood
    # https://math.stackexchange.com/questions/344487/log-likelihood-of-a-realization-of-a-poisson-process
    # Sum log lambdas across all time points
    if mask is None:
        poisson_log_l = torch.sum(info["log_lambda_y"], 2) - info["int_lambda"]
        # Sum over data dims
        poisson_log_l = torch.mean(poisson_log_l, -1)
    else:
        # Compute likelihood of the data under the predictions
        truth_repeated = truth.repeat(pred_y.size(0), 1, 1, 1)
        mask_repeated = mask.repeat(pred_y.size(0), 1, 1, 1)

        # Compute the likelihood per patient and per attribute so that we don't priorize patients with more measurements
        int_lambda = info["int_lambda"]

        def f(log_lam, data, indices):
            return poisson_log_likelihood(log_lam, data, indices, int_lambda)

        poisson_log_l = compute_masked_likelihood(
            info["log_lambda_y"], truth_repeated, mask_repeated, f
        )
        poisson_log_l = poisson_log_l.permute(1, 0)
        # Take mean over n_traj
        # poisson_log_l = torch.mean(poisson_log_l, 1)

    # poisson_log_l shape: [n_traj_samples, n_traj]
    return poisson_log_l
