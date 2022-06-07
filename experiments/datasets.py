###########################
# Neural Laplace: Learning diverse classes of differential equations in the Laplace domain
# Author: Samuel Holt
###########################
import shelve
from functools import partial
import numpy as np
import scipy.io as sio
import torch
from ddeint import ddeint
from torch.utils.data import DataLoader
from tqdm import tqdm

from torchlaplace.data_utils import (
    basic_collate_fn
)

from pathlib import Path
local_path = Path(__file__).parent

# DE Datasets
def dde_ramp_loading_time_sol(device, double=False, trajectories_to_sample=100, t_nsamples=200):
    t_end = 20.0
    t_begin = t_end / t_nsamples
    if double:
        ti = torch.linspace(t_begin, t_end, t_nsamples).to(device).double()
    else:
        ti = torch.linspace(t_begin, t_end, t_nsamples).to(device)


    def sampler(t, x0=0):
        ans = torch.ones_like(t) * x0 * torch.cos(2 * t)
        ans += (
            (1 / 5.0)
            * ((5 <= t) * (t < 10)).float()
            * (1.0 / 4.0)
            * ((t - 5) - 0.5 * torch.sin(2 * (t - 5)))
        )
        ans += (
            (1 / 5.0)
            * (10 <= t).float()
            * (1.0 / 4.0)
            * (
                (t - 5)
                - (t - 10)
                - 0.5 * torch.sin(2 * (t - 5))
                + 0.5 * torch.sin(2 * (t - 10))
            )
        )
        return ans

    x0s = torch.linspace(0, 1 / 10, trajectories_to_sample)
    trajs = []
    for x0 in x0s:
        trajs.append(sampler(ti, x0))
    y = torch.stack(trajs)
    trajectories = y.view(trajectories_to_sample, -1, 1)
    return trajectories, ti


def stiffvdp(device, double=False, trajectories_to_sample=100, one_dim=True):
    mat_contents = sio.loadmat(local_path / "data/vdp_all.mat")
    tm = mat_contents["t_samp"].ravel()
    trajs = mat_contents["all"]
    if double:
        trajs = torch.from_numpy(trajs).to(device).double()
    else:
        trajs = torch.from_numpy(trajs).to(torch.float32).to(device)
    trajectories = torch.transpose(trajs, 0, 2)
    trajectories = torch.transpose(trajectories, 1, 2)

    trajectories = trajectories[:trajectories_to_sample]

    t_scale = 20.0 / tm[-1]
    t = t_scale * tm
    if double:
        t = torch.from_numpy(t).to(device).double()
    else:
        t = torch.from_numpy(t).to(torch.float32).to(device)

    return trajectories, t


def integro_de(device, double=False, trajectories_to_sample=100, t_nsamples=200):
    t_end = 4.0
    t_begin = t_end / t_nsamples
    if double:
        ti = torch.linspace(t_begin, t_end, t_nsamples).to(device).double()
    else:
        ti = torch.linspace(t_begin, t_end, t_nsamples).to(device)


    def sampler(t, x0=0):
        return (
            (1 / 4.0)
            * torch.exp((-1 - 2j) * t)
            * (
                (2 * x0 + (x0 - 1) * 1j) * torch.exp(4 * 1j * t)
                + (2 * x0 - (x0 - 1) * 1j)
            )
        ).real

    x0s = torch.linspace(0, 1, trajectories_to_sample)
    trajs = []
    for x0 in x0s:
        trajs.append(sampler(ti, x0))
    y = torch.stack(trajs)
    trajectories = y.view(trajectories_to_sample, -1, 1)
    return trajectories, ti


def spiral_dde(device, double=False, trajectories_to_sample=100, t_nsamples=200):
    def model(XY, t, d):
        x, y = XY(t)
        xd, yd = XY(t - d)
        return np.array(
            [-np.tanh(x + xd) + np.tanh(y + yd), -np.tanh(x + xd) - np.tanh(y + yd),]
        )

    subsample_to_points = t_nsamples
    compute_points = 1000
    tt = np.linspace(20 / compute_points, 20, compute_points)
    sample_step = int(compute_points / subsample_to_points)
    trajectories_list = []

    evaluate_points = int(np.floor(np.sqrt(trajectories_to_sample)))
    x0s1d = np.linspace(-2, 2, evaluate_points)
    try:
        with shelve.open("datasets") as db:
            trajectories = db[f"spiral_dde_trajectories_{evaluate_points}"]
    except KeyError:
        for x0 in tqdm(x0s1d):
            for y0 in x0s1d:
                yy = ddeint(model, lambda t: np.array([x0, y0]), tt, fargs=(2.5,))
                trajectories_list.append(yy)
        trajectories = np.stack(trajectories_list)
        with shelve.open("datasets") as db:
            db[f"spiral_dde_trajectories_{evaluate_points}"] = trajectories
    trajectoriesn = trajectories[:, ::sample_step]
    tt = tt[::sample_step]
    if double:
        trajectories = torch.from_numpy(trajectoriesn).to(device).double()
        t = torch.from_numpy(tt).to(device).double()
    else:
        trajectories = torch.from_numpy(trajectoriesn).to(torch.float32).to(device)
        t = torch.from_numpy(tt).to(torch.float32).to(device)
    return trajectories, t


def lotka_volterra_system_with_delay(device, double=False, trajectories_to_sample=100, t_nsamples=200):
    def model(Y, t, d):
        x, y = Y(t)
        xd, yd = Y(t - d)
        return np.array([0.5 * x * (1 - yd), -0.5 * y * (1 - xd)])

    subsample_to_points = t_nsamples
    compute_points = 1000
    tt = np.linspace(2, 30, compute_points)
    sample_step = int(compute_points / subsample_to_points)
    trajectories_list = []

    evaluate_points = int(np.floor(np.sqrt(trajectories_to_sample)))
    x0s1d = np.linspace(0.1, 2, evaluate_points)
    try:
        with shelve.open("datasets") as db:
            trajectories = db[
                f"lotka_volterra_system_with_delay_trajectories_{evaluate_points}"
            ]
    except KeyError:
        for x0 in tqdm(x0s1d):
            for y0 in x0s1d:
                yy = ddeint(model, lambda t: np.array([x0, y0]), tt, fargs=(0.1,))
                trajectories_list.append(yy)
        trajectories = np.stack(trajectories_list)
        with shelve.open("datasets") as db:
            db[
                f"lotka_volterra_system_with_delay_trajectories_{evaluate_points}"
            ] = trajectories
    trajectoriesn = trajectories[:, ::sample_step]
    tt = tt[::sample_step]
    if double:
        trajectories = torch.from_numpy(trajectoriesn).to(device).double()
        t = torch.from_numpy(tt).to(device).double()
    else:
        trajectories = torch.from_numpy(trajectoriesn).to(torch.float32).to(device)
        t = torch.from_numpy(tt).to(torch.float32).to(device)
    return trajectories, t


def mackey_glass_dde_long_term_dep(device, double=False, trajectories_to_sample=100, t_nsamples=200):
    n = 10
    beta = 0.25
    gamma = 0.1
    THIRD = 50
    tau = THIRD * 2
    compute_points = 2000
    tt = np.linspace(0, THIRD, compute_points // 2)
    nt = np.linspace(-THIRD * 2, 0, compute_points // 2)
    sample_step = int(compute_points / t_nsamples)
    t_end = 20.0
    t_begin = t_end / t_nsamples
    if double:
        ti = torch.linspace(t_begin, t_end, t_nsamples).to(device).double()
    else:
        ti = torch.linspace(t_begin, t_end, t_nsamples).to(device)


    def model(Y, t, d):
        return beta * (Y(t - d) / (1 + Y(t - d) ** n)) - gamma * Y(t)

    def g(t, on_threshold):
        if t > -on_threshold:
            return 1.0
        else:
            return 0.0

    thres = THIRD * 2
    yp = ddeint(model, partial(g, thres), tt, fargs=(tau,))
    yy = [y if isinstance(y, float) else y[0] for y in yp]
    ny = np.array([g(t, thres) for t in nt])
    ya = np.concatenate((ny, yy))

    on_thresholds = np.linspace(THIRD * 2, 0, trajectories_to_sample)
    trajs = []
    for thres in tqdm(on_thresholds):
        yp = ddeint(model, partial(g, thres), tt, fargs=(tau,))
        yy = [y if isinstance(y, float) else y[0] for y in yp]
        ny = np.array([g(t, thres) for t in nt])
        ya = np.concatenate((ny, yy))
        trajs.append(ya[::sample_step])
    trajectoriesn = np.stack(trajs)
    if double:
        trajectories = torch.from_numpy(trajectoriesn).to(device).double()
    else:
        trajectories = torch.from_numpy(trajectoriesn).to(torch.float32).to(device)
    return trajectories.view(trajectories_to_sample, -1, 1), ti


# Waveform datasets
def sine(device, double=False, trajectories_to_sample=100, t_nsamples=200):
    t_end = 20.0
    t_begin = t_end / t_nsamples
    if double:
        ti = torch.linspace(t_begin, t_end, t_nsamples).to(device).double()
    else:
        ti = torch.linspace(t_begin, t_end, t_nsamples).to(device)

    def sampler(t, x0=0):
        return torch.sin(t + x0)

    x0s = torch.linspace(0, 2 * torch.pi, trajectories_to_sample)
    trajs = []
    for x0 in x0s:
        trajs.append(sampler(ti, x0))
    y = torch.stack(trajs)
    trajectories = y.view(trajectories_to_sample, -1, 1)
    return trajectories, ti


def square(device, double=False, trajectories_to_sample=100, t_nsamples=200):
    t_end = 20.0
    t_begin = t_end / t_nsamples
    if double:
        ti = torch.linspace(t_begin, t_end, t_nsamples).to(device).double()
    else:
        ti = torch.linspace(t_begin, t_end, t_nsamples).to(device)

    def sampler(t, x0=0):
        return (1 - torch.floor((t + x0) / torch.pi) % 2) * 2

    x0s = torch.linspace(0, 2 * torch.pi, trajectories_to_sample)
    trajs = []
    for x0 in x0s:
        trajs.append(sampler(ti, x0))
    y = torch.stack(trajs)
    trajectories = y.view(trajectories_to_sample, -1, 1)
    return trajectories, ti


def sawtooth(device, double=False, trajectories_to_sample=100, t_nsamples=200):
    t_end = 20.0
    t_begin = t_end / t_nsamples
    if double:
        ti = torch.linspace(t_begin, t_end, t_nsamples).to(device).double()
    else:
        ti = torch.linspace(t_begin, t_end, t_nsamples).to(device)

    def sampler(t, x0=0):
        return (t + x0) / (2 * torch.pi) - torch.floor((t + x0) / (2 * torch.pi))

    x0s = torch.linspace(0, 2 * torch.pi, trajectories_to_sample)
    trajs = []
    for x0 in x0s:
        trajs.append(sampler(ti, x0))
    y = torch.stack(trajs)
    trajectories = y.view(trajectories_to_sample, -1, 1)
    return trajectories, ti


def generate_data_set(
    name,
    device,
    double=False,
    batch_size=128,
    extrap=0,
    trajectories_to_sample=100,
    percent_missing_at_random=0.0,
    normalize=True,
    test_set_out_of_distribution=False,
    noise_std=None,
    t_nsamples=200,
    observe_step=1,
    predict_step=1,
):
    if name == "dde_ramp_loading_time_sol":
        trajectories, t = dde_ramp_loading_time_sol(device, double, trajectories_to_sample, t_nsamples)
    elif name == "spiral_dde":
        trajectories, t = spiral_dde(device, double, trajectories_to_sample, t_nsamples)
    elif name == "stiffvdp":
        trajectories, t = stiffvdp(device, double, trajectories_to_sample)
    elif name == "integro_de":
        trajectories, t = integro_de(device, double, trajectories_to_sample, t_nsamples)
    elif name == "mackey_glass_dde_long_term_dep":
        trajectories, t = mackey_glass_dde_long_term_dep(device, double, 
            trajectories_to_sample, t_nsamples
        )
    elif name == "lotka_volterra_system_with_delay":
        trajectories, t = lotka_volterra_system_with_delay(device, double, 
            trajectories_to_sample, t_nsamples
        )
    elif name == "sine":
        trajectories, t = sine(device, double, trajectories_to_sample, t_nsamples)
    elif name == "square":
        trajectories, t = square(device, double, trajectories_to_sample, t_nsamples)
    elif name == "sawtooth":
        trajectories, t = sawtooth(device, double, trajectories_to_sample, t_nsamples)
    else:
        raise ValueError("Unknown Dataset To Test")

    if not extrap:
        bool_mask = torch.FloatTensor(*trajectories.shape).uniform_() < (
            1.0 - percent_missing_at_random
        )
        if double:
            float_mask = (bool_mask).float().double().to(device)
        else:
            float_mask = (bool_mask).float().to(device)
        trajectories = float_mask * trajectories

    # normalize
    if normalize:
        samples = trajectories.shape[0]
        dim = trajectories.shape[2]
        traj = (
            torch.reshape(trajectories, (-1, dim))
            - torch.reshape(trajectories, (-1, dim)).mean(0)
        ) / torch.reshape(trajectories, (-1, dim)).std(0)
        trajectories = torch.reshape(traj, (samples, -1, dim))

    if noise_std:
        trajectories += torch.randn(trajectories.shape).to(device) * noise_std

    train_split = int(0.8 * trajectories.shape[0])
    test_split = int(0.9 * trajectories.shape[0])
    if test_set_out_of_distribution:
        train_trajectories = trajectories[:train_split, :, :]
        val_trajectories = trajectories[train_split:test_split, :, :]
        test_trajectories = trajectories[test_split:, :, :]
    else:
        traj_index = torch.randperm(trajectories.shape[0])
        train_trajectories = trajectories[traj_index[:train_split], :, :]
        val_trajectories = trajectories[traj_index[train_split:test_split], :, :]
        test_trajectories = trajectories[traj_index[test_split:], :, :]

    test_plot_traj = test_trajectories[0, :, :]

    input_dim = train_trajectories.shape[2]
    output_dim = input_dim

    dltrain = DataLoader(
        train_trajectories,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: basic_collate_fn(
            batch,
            t,
            data_type="train",
            extrap=extrap,
            observe_step=observe_step,
            predict_step=predict_step,
        ),
    )
    dlval = DataLoader(
        val_trajectories,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: basic_collate_fn(
            batch,
            t,
            data_type="test",
            extrap=extrap,
            observe_step=observe_step,
            predict_step=predict_step,
        ),
    )
    dltest = DataLoader(
        test_trajectories,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: basic_collate_fn(
            batch,
            t,
            data_type="test",
            extrap=extrap,
            observe_step=observe_step,
            predict_step=predict_step,
        ),
    )
    return (
        input_dim,
        output_dim,
        dltrain,
        dlval,
        dltest,
        test_plot_traj,
        t,
        test_trajectories,
    )
