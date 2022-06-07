###########################
# Neural Laplace: Learning diverse classes of differential equations in the Laplace domain
# Author: Samuel Holt
###########################
import logging
from copy import deepcopy
from time import time

import numpy as np
import torch

logger = logging.getLogger()

def train_and_test(
    system,
    dltrain,
    dlval,
    dltest,
    optim,
    device,
    scheduler=None,
    epochs=1000,
    patience=None,
    gradient_clip=1,
):
    # Model is an super class of the actual model used - to give training methods,
    # Training loop parameters
    if not patience:
        patience = epochs
    best_loss = float("inf")
    waiting = 0
    durations = []
    train_losses = []
    val_losses = []
    train_nfes = []
    epoch_num = []

    for epoch in range(epochs):
        iteration = 0
        epoch_train_loss_it_cum = 0
        epoch_nfe_cum = 0

        system.model.train()
        start_time = time()

        for batch in dltrain:
            # Single training step
            optim.zero_grad()
            train_loss = system.training_step(batch)
            iteration_nfes = system._get_and_reset_nfes()
            train_loss.backward()
            # Optional gradient clipping
            torch.nn.utils.clip_grad_norm_(system.model.parameters(), gradient_clip)
            optim.step()
            epoch_train_loss_it_cum += train_loss.item()
            epoch_nfe_cum += iteration_nfes

            # logger.info(
            #     f'[epoch={epoch+1:04d}|iter={iteration+1:04d}] train_loss={train_loss:.5f}')
            iteration += 1
        epoch_train_loss = epoch_train_loss_it_cum / iteration
        epoch_nfes = epoch_nfe_cum / iteration

        epoch_duration = time() - start_time
        durations.append(epoch_duration)
        train_losses.append(epoch_train_loss)
        train_nfes.append(epoch_nfes)
        epoch_num.append(epoch)

        # Validation step
        system.model.eval()
        val_loss, val_mse = system.validation_step(dlval)
        val_loss, val_mse = val_loss.item(), val_mse.item()
        val_losses.append(val_loss)
        logger.info(
            "[epoch={}] epoch_duration={:.2f} | train_loss={}\t| val_loss={}\t| val_mse={}".format(
                epoch, epoch_duration, epoch_train_loss, val_loss, val_mse
            )
        )

        # Learning rate scheduler
        if scheduler:
            scheduler.step()

        # Early stopping procedure
        if val_loss < best_loss:
            best_loss = val_loss
            best_model = deepcopy(system.model.state_dict())
            waiting = 0
        elif waiting > patience:
            break
        else:
            waiting += 1

    logger.info(f"epoch_duration_mean={np.mean(durations):.5f}")

    # Load best model
    system.model.load_state_dict(best_model)

    # Held-out test set step
    test_loss, test_mse = system.test_step(dltest)
    test_mse = test_mse.item()
    test_rmse = np.sqrt(test_mse)
    return (
        test_rmse,
        torch.Tensor(train_losses).to(device),
        torch.Tensor(train_nfes).to(device),
        torch.Tensor(epoch_num).to(device),
    )
