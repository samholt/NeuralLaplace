###########################
# Neural Laplace: Learning diverse classes of differential equations in the Laplace domain
# Author: Samuel Holt
###########################
import argparse
import logging
import pickle
from pathlib import Path
from time import strftime
import pandas as pd
import numpy as np
import torch
from datasets import generate_data_set

from baseline_models.neural_laplace import GeneralNeuralLaplace
from baseline_models.ode_models import GeneralLatentODE
from baseline_models.original_latent_ode import GeneralLatentODEOfficial
from baseline_models.original_ode_models import GeneralNODE
from utils import train_and_test

datasets = [
    "dde_ramp_loading_time_sol",
    "spiral_dde",
    "stiffvdp",
    "lotka_volterra_system_with_delay",
    "integro_de",
    "mackey_glass_dde_long_term_dep",
    "sine",
    "square",
    "sawtooth",
]

np.random.seed(999)
torch.random.manual_seed(999)

file_name = Path(__file__).stem

def experiment_with_all_baselines(
    dataset,
    batch_size,
    extrapolate,
    epochs,
    seed,
    run_number_of_seeds,
    learning_rate,
    ode_solver_method,
    trajectories_to_sample,
    time_points_to_sample,
    observe_step,
    predict_step,
    noise_std,
    normalize_dataset,
    encode_obs_time,
    latent_dim,
    s_recon_terms,
    patience,
    device,
    use_sphere_projection,
    ilt_algorithm
    ):
    # Compares against all baselines, returning a pandas DataFrame of the test RMSE extrapolation error with std across input seed runs
    # Also saves out training meta-data in a ./results folder (such as training loss array and NFE array against the epochs array)
    observe_samples = (time_points_to_sample // 2) // observe_step
    logger.info(f"Experimentally observing {observe_samples} samples")

    df_list_baseline_results = []

    for seed in range(seed, seed + run_number_of_seeds):
        torch.random.manual_seed(seed)

        Path("./results").mkdir(parents=True, exist_ok=True)
        path = f"./results/{path_run_name}-{seed}.pkl"

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
            dataset,
            device,
            double=True,
            batch_size=batch_size,
            trajectories_to_sample=trajectories_to_sample,
            extrap=extrapolate,
            normalize=normalize_dataset,
            noise_std=noise_std,
            t_nsamples=time_points_to_sample,
            observe_step=observe_step,
            predict_step=predict_step,
        )

        saved_dict = {}

        saved_dict["dataset"] = dataset
        saved_dict["trajectories_to_sample"] = trajectories_to_sample
        saved_dict["extrapolate"] = extrapolate
        saved_dict["normalize_dataset"] = normalize_dataset
        saved_dict["input_dim"] = input_dim
        saved_dict["output_dim"] = output_dim

        # Pre-save
        with open(path, "wb") as f:
            pickle.dump(saved_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

        for model_name, system in [
            (
                "Neural Laplace",
                GeneralNeuralLaplace(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    latent_dim=latent_dim,
                    hidden_units=42,
                    s_recon_terms=s_recon_terms,
                    use_sphere_projection=use_sphere_projection,
                    ilt_algorithm=ilt_algorithm,
                    encode_obs_time=encode_obs_time,
                ).to(device),
            ),
            (
                f"NODE ({ode_solver_method})",
                GeneralNODE(
                    obs_dim=input_dim,
                    nhidden=128,
                    method=ode_solver_method,
                    extrap=extrapolate,
                ).to(device),
            ),
            (
                f"ANODE ({ode_solver_method})",
                GeneralNODE(
                    obs_dim=input_dim,
                    nhidden=128,
                    method=ode_solver_method,
                    extrap=extrapolate,
                    augment_dim=1,
                ).to(device),
            ),
            (
                "Latent ODE (ODE enc.)",
                GeneralLatentODEOfficial(
                    input_dim, n_labels=1, obsrv_std=0.01, latents=2, hidden_units=40,
                ).to(device),
            ),
            (
                "Neural Flow Coupling",
                GeneralLatentODE(
                    input_dim,
                    model="flow",
                    flow_model="coupling",
                    hidden_dim=31,
                    hidden_layers=latent_dim,
                    latents=latent_dim,
                    n_classes=input_dim,
                ).to(device),
            ),
            (
                "Neural Flow ResNet",
                GeneralLatentODE(
                    input_dim,
                    model="flow",
                    flow_model="resnet",
                    hidden_dim=26,
                    hidden_layers=latent_dim,
                    latents=latent_dim,
                    n_classes=input_dim,
                ).to(device),
            ),
        ]:
            try:
                logger.info(f"Training & testing for : {model_name} \t | seed: {seed}")
                system.double()
                logger.info(
                    "num_params={}".format(
                        sum(p.numel() for p in system.model.parameters())
                    )
                )
                optimizer = torch.optim.Adam(
                    system.model.parameters(), lr=learning_rate
                )
                lr_scheduler_step = 20
                lr_decay = 0.5
                scheduler = None
                test_rmse, train_loss, train_nfes, train_epochs = train_and_test(
                    system,
                    dltrain,
                    dlval,
                    dltest,
                    optimizer,
                    device,
                    scheduler,
                    epochs=epochs,
                    patience=patience,
                )
                logger.info(f"Result: {model_name} - TEST RMSE: {test_rmse}")
                df_list_baseline_results.append({'method': model_name, 'test_rmse': test_rmse, 'seed': seed})
                saved_dict[model_name] = {
                    "test rmse": test_rmse,
                    "seed": seed,
                    "model_state_dict": system.model.state_dict(),
                    "train_loss": train_loss.detach().cpu().numpy(),
                    "train_nfes": train_nfes.detach().cpu().numpy(),
                    "train_epochs": train_epochs.detach().cpu().numpy(),
                }
                # Checkpoint
                with open(path, "wb") as f:
                    pickle.dump(saved_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                logger.error(e)
                logger.error(f"Error for model: {model_name}")
                raise e
        path = f"./results/{path_run_name}-{seed}.pkl"
        with open(path, "wb") as f:
            pickle.dump(saved_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Process results for experiment
    df_results = pd.DataFrame(df_list_baseline_results)
    test_rmse_df = df_results.groupby('method').agg(['mean', 'std'])['test_rmse']
    logger.info("Test RMSE of experiment")
    logger.info(test_rmse_df.style.to_latex())
    return test_rmse_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run all baselines for an experiment (including Neural Laplace)")
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="dde_ramp_loading_time_sol",
        help=f"Available datasets: {datasets}",
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--extrapolate", action="store_false")  # Default True
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run_number_of_seeds", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--ode_solver_method", type=str, default="euler")
    parser.add_argument("--trajectories_to_sample", type=int, default=1000)
    parser.add_argument("--time_points_to_sample", type=int, default=200)
    parser.add_argument("--observe_step", type=int, default=1)
    parser.add_argument("--predict_step", type=int, default=1)
    parser.add_argument("--noise_std", type=float, default=0.0)
    parser.add_argument("--normalize_dataset", action="store_false")  # Default True
    parser.add_argument("--encode_obs_time", action="store_true")  # Default False
    parser.add_argument("--latent_dim", type=int, default=2)
    parser.add_argument("--s_recon_terms", type=int, default=33)  # (ANGLE_SAMPLES * 2 + 1)
    parser.add_argument("--patience", nargs="?", type=int, const=500)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--use_sphere_projection", action="store_false")  # Default True
    parser.add_argument("--ilt_algorithm", type=str, default="fourier")
    args = parser.parse_args()

    assert args.dataset in datasets
    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

    Path("./logs").mkdir(parents=True, exist_ok=True)
    path_run_name = "{}-{}-{}".format(
        file_name, strftime("%Y%m%d-%H%M%S"), args.dataset
    )
    logging.basicConfig(
        format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(f"logs/{path_run_name}_log.txt"),
            logging.StreamHandler()
        ],
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )
    logger = logging.getLogger()

    logger.info(f"Using {device} device")
    test_rmse_df = experiment_with_all_baselines(
        args.dataset,
        args.batch_size,
        args.extrapolate,
        args.epochs,
        args.seed,
        args.run_number_of_seeds,
        args.learning_rate,
        args.ode_solver_method,
        args.trajectories_to_sample,
        args.time_points_to_sample,
        args.observe_step,
        args.predict_step,
        args.noise_std,
        args.normalize_dataset,
        args.encode_obs_time,
        args.latent_dim,
        args.s_recon_terms,
        args.patience,
        device,
        args.use_sphere_projection,
        args.ilt_algorithm)

        
