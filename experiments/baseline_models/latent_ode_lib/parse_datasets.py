# pylint: skip-file
# pytype: skip-file
###########################
# Latent ODEs for Irregularly-Sampled Time Series
# Author: Yulia Rubanova
###########################


import numpy as np
import torch
from sklearn import model_selection

# from generate_timeseries import Periodic_1d
from torch.distributions import uniform
from torch.utils.data import DataLoader

from .utils import inf_generator, split_and_subsample_batch, split_train_test

# from mujoco_physics import HopperPhysics
# from physionet import PhysioNet, variable_time_collate_fn, get_data_min_max
# from person_activity import PersonActivity, variable_time_collate_fn_activity

#####################################################################################################


def sine(trajectories_to_sample, device):
  t_end = 20.0
  t_nsamples = 200
  t_begin = t_end / t_nsamples
  t = torch.linspace(t_begin, t_end, t_nsamples).to(device).double()
  y = torch.sin(t)
  trajectories = y.view(1, -1, 1).repeat(trajectories_to_sample, 1, 1)
  return trajectories, t


def dde_ramp_loading_time_sol(trajectories_to_sample, device):
  t_end = 20.0
  t_nsamples = 200
  t_begin = t_end / t_nsamples
  ti = torch.linspace(t_begin, t_end, t_nsamples).to(device).double()
  result = []
  for t in ti:
    if t < 5:
      result.append(0)
    elif 5 <= t < 10:
      result.append((1.0 / 4.0) * ((t - 5) - 0.5 * torch.sin(2 * (t - 5))))
    elif 10 <= t:
      result.append(
          (1.0 / 4.0) *
          ((t - 5) -
           (t - 10) - 0.5 * torch.sin(2 * (t - 5)) + 0.5 * torch.sin(2 *
                                                                     (t - 10))))
  y = torch.Tensor(result).to(device).double() / 5.0
  trajectories = y.view(1, -1, 1).repeat(trajectories_to_sample, 1, 1)
  return trajectories, ti


def parse_datasets(args, device):

  def basic_collate_fn(batch,
                       time_steps,
                       args=args,
                       device=device,
                       data_type="train"):
    batch = torch.stack(batch)
    data_dict = {"data": batch, "time_steps": time_steps}

    data_dict = split_and_subsample_batch(data_dict, args, data_type=data_type)
    return data_dict

  dataset_name = args.dataset

  n_total_tp = args.timepoints + args.extrap
  max_t_extrap = args.max_t / args.timepoints * n_total_tp
  if dataset_name == "sine" or dataset_name == "dde_ramp_loading_time_sol":
    trajectories_to_sample = 1000
    if dataset_name == "sine":
      trajectories, t = sine(trajectories_to_sample, device)
    elif dataset_name == "dde_ramp_loading_time_sol":
      trajectories, t = dde_ramp_loading_time_sol(trajectories_to_sample,
                                                  device)

    # # Normalise
    # samples = trajectories.shape[0]
    # dim = trajectories.shape[2]
    # traj = (trajectories.view(-1, dim) - trajectories.view(-1,
    #         dim).mean(0)) / trajectories.view(-1, dim).std(0)
    # trajectories = torch.reshape(traj, (samples, -1, dim))

    traj_index = torch.randperm(trajectories.shape[0])
    train_split = int(0.8 * trajectories.shape[0])
    test_split = int(0.9 * trajectories.shape[0])
    train_trajectories = trajectories[traj_index[:train_split], :, :]
    test_trajectories = trajectories[traj_index[test_split:], :, :]

    test_trajectories[0, :, :]

    input_dim = train_trajectories.shape[2]
    batch_size = 128

    train_dataloader = DataLoader(
        train_trajectories,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: basic_collate_fn(batch, t, data_type="train"),
    )
    test_dataloader = DataLoader(
        test_trajectories,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: basic_collate_fn(batch, t, data_type="test"),
    )

    data_objects = {
        "dataset_obj": "",
        "train_dataloader": inf_generator(train_dataloader),
        "test_dataloader": inf_generator(test_dataloader),
        "input_dim": input_dim,
        "n_train_batches": len(train_dataloader),
        "n_test_batches": len(test_dataloader),
    }
    return data_objects

  ##################################################################
  # MuJoCo dataset
  if dataset_name == "hopper":
    dataset_obj = HopperPhysics(root="data",
                                download=True,
                                generate=False,
                                device=device)
    dataset = dataset_obj.get_dataset()[:args.n]
    dataset = dataset.to(device).double()

    n_tp_data = dataset[:].shape[1]

    # Time steps that are used later on for exrapolation
    time_steps = (torch.arange(start=0, end=n_tp_data,
                               step=1).float().to(device).double())
    time_steps = time_steps / len(time_steps)

    dataset = dataset.to(device).double()
    time_steps = time_steps.to(device).double()

    if not args.extrap:
      # Creating dataset for interpolation
      # sample time points from different parts of the timeline,
      # so that the model learns from different parts of hopper trajectory
      n_traj = len(dataset)
      n_tp_data = dataset.shape[1]
      n_reduced_tp = args.timepoints

      # sample time points from different parts of the timeline,
      # so that the model learns from different parts of hopper trajectory
      start_ind = np.random.randint(0,
                                    high=n_tp_data - n_reduced_tp + 1,
                                    size=n_traj)
      end_ind = start_ind + n_reduced_tp
      sliced = []
      for i in range(n_traj):
        sliced.append(dataset[i, start_ind[i]:end_ind[i], :])
      dataset = torch.stack(sliced).to(device).double()
      time_steps = time_steps[:n_reduced_tp]

    # Split into train and test by the time sequences
    train_y, test_y = split_train_test(dataset, train_fraq=0.8)

    n_samples = len(dataset)
    input_dim = dataset.size(-1)

    batch_size = min(args.batch_size, args.n)
    train_dataloader = DataLoader(
        train_y,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: basic_collate_fn(
            batch, time_steps, data_type="train"),
    )
    test_dataloader = DataLoader(
        test_y,
        batch_size=n_samples,
        shuffle=False,
        collate_fn=lambda batch: basic_collate_fn(
            batch, time_steps, data_type="test"),
    )

    data_objects = {
        "dataset_obj": dataset_obj,
        "train_dataloader": inf_generator(train_dataloader),
        "test_dataloader": inf_generator(test_dataloader),
        "input_dim": input_dim,
        "n_train_batches": len(train_dataloader),
        "n_test_batches": len(test_dataloader),
    }
    return data_objects

  ##################################################################
  # Physionet dataset

  if dataset_name == "physionet":
    train_dataset_obj = PhysioNet(
        "data/physionet",
        train=True,
        quantization=args.quantization,
        download=True,
        n_samples=min(10000, args.n),
        device=device,
    )
    # Use custom collate_fn to combine samples with arbitrary time observations.
    # Returns the dataset along with mask and time steps
    test_dataset_obj = PhysioNet(
        "data/physionet",
        train=False,
        quantization=args.quantization,
        download=True,
        n_samples=min(10000, args.n),
        device=device,
    )

    # Combine and shuffle samples from physionet Train and physionet Test
    total_dataset = train_dataset_obj[:len(train_dataset_obj)]

    if not args.classif:
      # Concatenate samples from original Train and Test sets
      # Only 'training' physionet samples are have labels. Therefore, if we do classifiction task, we don't need physionet 'test' samples.
      total_dataset = total_dataset + test_dataset_obj[:len(test_dataset_obj)]

    # Shuffle and split
    train_data, test_data = model_selection.train_test_split(total_dataset,
                                                             train_size=0.8,
                                                             random_state=42,
                                                             shuffle=True)

    record_id, tt, vals, mask, labels = train_data[0]

    n_samples = len(total_dataset)
    input_dim = vals.size(-1)

    batch_size = min(min(len(train_dataset_obj), args.batch_size), args.n)
    data_min, data_max = get_data_min_max(total_dataset)

    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: variable_time_collate_fn(
            batch,
            args,
            device,
            data_type="train",
            data_min=data_min,
            data_max=data_max,
        ),
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=n_samples,
        shuffle=False,
        collate_fn=lambda batch: variable_time_collate_fn(
            batch,
            args,
            device,
            data_type="test",
            data_min=data_min,
            data_max=data_max,
        ),
    )

    attr_names = train_dataset_obj.params
    data_objects = {
        "dataset_obj": train_dataset_obj,
        "train_dataloader": inf_generator(train_dataloader),
        "test_dataloader": inf_generator(test_dataloader),
        "input_dim": input_dim,
        "n_train_batches": len(train_dataloader),
        "n_test_batches": len(test_dataloader),
        "attr": attr_names,  # optional
        "classif_per_tp": False,  # optional
        "n_labels": 1,
    }  # optional
    return data_objects

  ##################################################################
  # Human activity dataset

  if dataset_name == "activity":
    n_samples = min(10000, args.n)
    dataset_obj = PersonActivity("data/PersonActivity",
                                 download=True,
                                 n_samples=n_samples,
                                 device=device)
    print(dataset_obj)
    # Use custom collate_fn to combine samples with arbitrary time observations.
    # Returns the dataset along with mask and time steps

    # Shuffle and split
    train_data, test_data = model_selection.train_test_split(dataset_obj,
                                                             train_size=0.8,
                                                             random_state=42,
                                                             shuffle=True)

    train_data = [
        train_data[i]
        for i in np.random.choice(len(train_data), len(train_data))
    ]
    test_data = [
        test_data[i] for i in np.random.choice(len(test_data), len(test_data))
    ]

    record_id, tt, vals, mask, labels = train_data[0]
    input_dim = vals.size(-1)

    batch_size = min(min(len(dataset_obj), args.batch_size), args.n)
    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: variable_time_collate_fn_activity(
            batch, args, device, data_type="train"),
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=n_samples,
        shuffle=False,
        collate_fn=lambda batch: variable_time_collate_fn_activity(
            batch, args, device, data_type="test"),
    )

    data_objects = {
        "dataset_obj": dataset_obj,
        "train_dataloader": inf_generator(train_dataloader),
        "test_dataloader": inf_generator(test_dataloader),
        "input_dim": input_dim,
        "n_train_batches": len(train_dataloader),
        "n_test_batches": len(test_dataloader),
        "classif_per_tp": True,  # optional
        "n_labels": labels.size(-1),
    }

    return data_objects

  ########### 1d datasets ###########

  # Sampling args.timepoints time points in the interval [0, args.max_t]
  # Sample points for both training sequence and explapolation (test)
  distribution = uniform.Uniform(torch.Tensor([0.0]),
                                 torch.Tensor([max_t_extrap]))
  time_steps_extrap = distribution.sample(torch.Size([n_total_tp - 1]))[:, 0]
  time_steps_extrap = torch.cat((torch.Tensor([0.0]), time_steps_extrap))
  time_steps_extrap = torch.sort(time_steps_extrap)[0]

  dataset_obj = None
  ##################################################################
  # Sample a periodic function
  if dataset_name == "periodic":
    dataset_obj = Periodic_1d(
        init_freq=None,
        init_amplitude=1.0,
        final_amplitude=1.0,
        final_freq=None,
        z0=1.0,
    )

  ##################################################################

  if dataset_obj is None:
    raise Exception(f"Unknown dataset: {dataset_name}")

  dataset = dataset_obj.sample_traj(time_steps_extrap,
                                    n_samples=args.n,
                                    noise_weight=args.noise_weight)

  # Process small datasets
  dataset = dataset.to(device).double()
  time_steps_extrap = time_steps_extrap.to(device).double()

  train_y, test_y = split_train_test(dataset, train_fraq=0.8)

  n_samples = len(dataset)
  input_dim = dataset.size(-1)

  batch_size = min(args.batch_size, args.n)
  train_dataloader = DataLoader(
      train_y,
      batch_size=batch_size,
      shuffle=False,
      collate_fn=lambda batch: basic_collate_fn(
          batch, time_steps_extrap, data_type="train"),
  )
  test_dataloader = DataLoader(
      test_y,
      batch_size=args.n,
      shuffle=False,
      collate_fn=lambda batch: basic_collate_fn(
          batch, time_steps_extrap, data_type="test"),
  )

  data_objects = {  # "dataset_obj": dataset_obj,
      "train_dataloader": inf_generator(train_dataloader),
      "test_dataloader": inf_generator(test_dataloader),
      "input_dim": input_dim,
      "n_train_batches": len(train_dataloader),
      "n_test_batches": len(test_dataloader),
  }

  return data_objects
