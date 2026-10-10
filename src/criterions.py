"""Subspace-Net 
Details
----------
Name: criterions.py
Authors: D. H. Shmuel
Created: 01/10/21
Edited: 03/06/23

Purpose:
--------
The purpose of this script is to define and document several loss functions (RMSPELoss and MSPELoss)
and a helper function (permute_prediction) for calculating the Root Mean Square Periodic Error (RMSPE)
and Mean Square Periodic Error (MSPE) between predicted values and target values.
The script also includes a utility function RMSPE and MSPE that calculates the RMSPE and MSPE values
for numpy arrays.

This script includes the following Classes anf functions:

* permute_prediction: A function that generates all possible permutations of a given prediction tensor.
* RMSPELoss (class): A custom PyTorch loss function that calculates the RMSPE loss between predicted values
    and target values. It inherits from the nn.Module class and overrides the forward method to perform
    the loss computation.
* MSPELoss (class): A custom PyTorch loss function that calculates the MSPE loss between predicted values
  and target values. It inherits from the nn.Module class and overrides the forward method to perform the loss computation.
* RMSPE (function): A function that calculates the RMSPE value between the DOA predictions and target DOA values for numpy arrays.
* MSPE (function): A function that calculates the MSPE value between the DOA predictions and target DOA values for numpy arrays.
* set_criterions(function): Set the loss criteria based on the criterion name.

"""

import numpy as np
import torch.nn as nn
import torch
from itertools import permutations
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu");

def permute_prediction_batched(predictions: torch.Tensor):
    """All permutations of every prediction vector in a batch, as one gather.

    Batched equivalent of :func:`permute_prediction`, which is called once per sample inside a
    Python loop. The permutation index table is built once per call and used to index the whole
    batch at once.

    Args:
    -----
        predictions (torch.Tensor): shape [Batch size, M].

    Returns:
    --------
        torch.Tensor: shape [Batch size, M!, M], where entry [b, p] is
        ``predictions[b]`` reordered by the p-th permutation.
    """
    m = predictions.shape[-1]
    perms = torch.tensor(
        list(permutations(range(m), m)), dtype=torch.long, device=predictions.device
    )
    return predictions[:, perms]


def permute_prediction(prediction: torch.Tensor):
    """
    Generates all the available permutations of the given prediction tensor.

    Args:
        prediction (torch.Tensor): The input tensor for which permutations are generated.

    Returns:
        torch.Tensor: A tensor containing all the permutations of the input tensor.

    Examples:
        >>> prediction = torch.tensor([1, 2, 3])
        >>>> permute_prediction(prediction)
            torch.tensor([[1, 2, 3],
                          [1, 3, 2],
                          [2, 1, 3],
                          [2, 3, 1],
                          [3, 1, 2],
                          [3, 2, 1]])
        
    """
    torch_perm_list = []
    for p in list(permutations(range(prediction.shape[0]),prediction.shape[0])):
        torch_perm_list.append(prediction.index_select( 0, torch.tensor(list(p), dtype = torch.int64).to(device)))
    predictions = torch.stack(torch_perm_list, dim = 0)
    return predictions

def trainable_device(module, reference: torch.Tensor) -> torch.device:
    """Device to put criterion temporaries on: wherever the predictions already live.

    The module-level ``device`` below is fixed at import time (``cuda:0`` whenever any GPU is
    visible), so any script that runs on a different card -- ``CUDA_VISIBLE_DEVICES``, or the
    ``--device`` flag of the benchmark scripts, which patch ``src.utils.device`` after import --
    would otherwise move predictions to ``cuda:0`` and then compare them against a target that
    is still on ``cuda:1``::

        RuntimeError: Expected all tensors to be on the same device, but found at least two
        devices, cuda:0 and cuda:1!

    Instead of trusting the import-time constant, follow the tensor that is handed in. The
    chosen device is cached on the module purely to avoid re-deciding it on every call.
    """
    cached = getattr(module, "_followed_device", None)
    if cached is not None and cached == reference.device:
        return cached
    module._followed_device = reference.device
    return reference.device


class RMSPELoss(nn.Module):
    """Root Mean Square Periodic Error (RMSPE) loss function.
    This loss function calculates the RMSPE between the predicted values and the target values.
    The predicted values and target values are expected to be in radians.

    Args:
        None

    Attributes:
        None

    Methods:
        forward(doa_predictions: torch.Tensor, doa: torch.Tensor) -> torch.Tensor:
            Computes the RMSPE loss between the predictions and target values.

    Example:
        criterion = RMSPELoss()
        predictions = torch.tensor([0.5, 1.2, 2.0])
        targets = torch.tensor([0.8, 1.5, 1.9])
        loss = criterion(predictions, targets)
    """
    def __init__(self):
        super(RMSPELoss, self).__init__()
    def forward(self, doa_predictions: torch.Tensor, doa: torch.Tensor):
        """Compute the RMSPE loss between the predictions and target values.
        The forward method takes two input tensors: doa_predictions and doa.
        The predicted values and target values are expected to be in radians.
        The method iterates over the batch dimension and calculates the RMSPE loss for each sample in the batch.
        It utilizes the permute_prediction function to generate all possible permutations of the predicted values
        to consider all possible alignments. For each permutation, it calculates the error between the prediction
        and target values, applies modulo pi to ensure the error is within the range [-pi/2, pi/2], and then calculates the RMSPE.
        The minimum RMSPE value among all permutations is selected for each sample.
        Finally, the method sums up the RMSPE values for all samples in the batch and returns the result as the computed loss.

        Args:
            doa_predictions (torch.Tensor): Predicted values tensor of shape (batch_size, num_predictions).
            doa (torch.Tensor): Target values tensor of shape (batch_size, num_targets).

        Returns:
            torch.Tensor: The computed RMSPE loss.

        Raises:
            None

        Note:
        -----
            Batched re-implementation of the original per-sample loop. The loop issued one
            ``permute_prediction``, ``torch.linalg.norm``, ``torch.stack`` and ``torch.min`` per
            sample per permutation -- i.e. ``batch * M!`` tiny reductions, each preceded by a
            host/device synchronization from the ``.item()`` inside ``torch.min``. On a training
            step of batch 512 that made this criterion the single most expensive part of the step,
            several times the cost of the whole forward pass. Every operation below is vectorized
            over the batch; the arithmetic is unchanged, so the returned loss and the gradient are
            bit-identical to the loop (see ``bench_step_split.py`` and the repository notes).
        """
        # Follow the predictions rather than the import-time `cuda:0` constant, so that running
        # on a non-default card does not mix devices (see trainable_device above).
        target_device = trainable_device(self, doa_predictions)
        prediction_perm = permute_prediction_batched(doa_predictions).to(target_device)  # [B, M!, M]
        error = (((prediction_perm - doa.to(target_device).unsqueeze(1)) + (np.pi / 2)) % np.pi) - np.pi / 2
        rmspe_val = (1 / np.sqrt(doa.shape[-1])) * torch.linalg.norm(error, dim=-1)  # [B, M!]
        # Minimal error over all permutations, per sample, then the sum over the batch.
        return torch.min(rmspe_val, dim=1).values.sum()

class MSPELoss(nn.Module):
    """Mean Square Periodic Error (MSPE) loss function.
    This loss function calculates the MSPE between the predicted values and the target values.
    The predicted values and target values are expected to be in radians.

    Args:
        None

    Attributes:
        None

    Methods:
        forward(doa_predictions: torch.Tensor, doa: torch.Tensor) -> torch.Tensor:
            Computes the MSPE loss between the predictions and target values.

    Example:
        criterion = MSPELoss()
        predictions = torch.tensor([0.5, 1.2, 2.0])
        targets = torch.tensor([0.8, 1.5, 1.9])
        loss = criterion(predictions, targets)
    """
    def __init__(self):
        super(MSPELoss, self).__init__()
    def forward(self, doa_predictions: torch.Tensor, doa):
        """Compute the RMSPE loss between the predictions and target values.
        The forward method takes two input tensors: doa_predictions and doa.
        The predicted values and target values are expected to be in radians.
        The method iterates over the batch dimension and calculates the RMSPE loss for each sample in the batch.
        It utilizes the permute_prediction function to generate all possible permutations of the predicted values
        to consider all possible alignments. For each permutation, it calculates the error between the prediction
        and target values, applies modulo pi to ensure the error is within the range [-pi/2, pi/2], and then calculates the RMSPE.
        The minimum RMSPE value among all permutations is selected for each sample.
        Finally, the method sums up the RMSPE values for all samples in the batch and returns the result as the computed loss.

        Args:
            doa_predictions (torch.Tensor): Predicted values tensor of shape (batch_size, num_predictions).
            doa (torch.Tensor): Target values tensor of shape (batch_size, num_targets).

        Returns:
            torch.Tensor: The computed MSPE loss.

        Raises:
            None

        Note:
        -----
            Batched re-implementation of the original per-sample loop; see the note on
            :meth:`RMSPELoss.forward`. Same arithmetic, so the loss and gradient are unchanged.
        """
        target_device = trainable_device(self, doa_predictions)
        prediction_perm = permute_prediction_batched(doa_predictions).to(target_device)  # [B, M!, M]
        error = (((prediction_perm - doa.to(target_device).unsqueeze(1)) + (np.pi / 2)) % np.pi) - np.pi / 2
        rmspe_val = (1 / doa.shape[-1]) * (torch.linalg.norm(error, dim=-1) ** 2)  # [B, M!]
        return torch.min(rmspe_val, dim=1).values.sum()

def RMSPE(doa_predictions: np.ndarray, doa: np.ndarray):
    """
    Calculate the Root Mean Square Periodic Error (RMSPE) between the DOA predictions and target DOA values.

    Args:
        doa_predictions (np.ndarray): Array of DOA predictions.
        doa (np.ndarray): Array of target DOA values.

    Returns:
        float: The computed RMSPE value.

    Raises:
        None
    """
    rmspe_list = []
    for p in list(permutations(doa_predictions, len(doa_predictions))):
        p = np.array(p)
        doa = np.array(doa)
        # Calculate error with modulo pi
        error = (((p - doa) * np.pi / 180) + np.pi / 2) % np.pi - np.pi / 2
        # Calculate RMSE over all permutations
        rmspe_val = (1 / np.sqrt(len(p))) * np.linalg.norm(error)
        rmspe_list.append(rmspe_val)
    # Choose minimal error from all permutations
    return np.min(rmspe_list)

def MSPE(doa_predictions: np.ndarray, doa: np.ndarray):
    """Calculate the Mean Square Percentage Error (RMSPE) between the DOA predictions and target DOA values.

    Args:
        doa_predictions (np.ndarray): Array of DOA predictions.
        doa (np.ndarray): Array of target DOA values.

    Returns:
        float: The computed RMSPE value.

    Raises:
        None
    """
    rmspe_list = []
    for p in list(permutations(doa_predictions, len(doa_predictions))):
        p = np.array(p)
        doa = np.array(doa)
        # Calculate error with modulo pi
        error = (((p - doa) * np.pi / 180) + np.pi / 2) % np.pi - np.pi / 2
        # Calculate MSE over all permutations
        rmspe_val = (1 / len(p)) * (np.linalg.norm(error) ** 2)
        rmspe_list.append(rmspe_val)
    # Choose minimal error from all permutations
    return np.min(rmspe_list)

def set_criterions(criterion_name:str):
    """
    Set the loss criteria based on the criterion name.

    Parameters:
        criterion_name (str): Name of the criterion.

    Returns:
        criterion (nn.Module): Loss criterion for model evaluation.
        subspace_criterion (Callable): Loss criterion for subspace method evaluation.

    Raises:
        Exception: If the criterion name is not defined.
    """
    if criterion_name.startswith("rmse"):
        criterion = RMSPELoss()
        subspace_criterion = RMSPE
    elif criterion_name.startswith("mse"):
        criterion = MSPELoss()
        subspace_criterion = MSPE
    else:
        raise Exception(f"criterions.set_criterions: Criterion {criterion_name} is not defined")
    print(f"Loss measure = {criterion_name}")
    return criterion, subspace_criterion

if __name__ == "__main__":
    prediction = torch.tensor([1, 2, 3])
    print(permute_prediction(prediction))