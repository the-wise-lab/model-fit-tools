import matplotlib.pyplot as plt
import numpy as np
import os
from typing import List
import datetime


def plot_recovery(
    true: np.ndarray, 
    estimated:np.ndarray, 
    param_names:List[str]=None,
    show_correlation:bool=True,
    scale:float=1.0,
    save_path:str=None, 
    save_fname:str=None
):
    """
    Plots recovered parameter values against true ones. Used to determine how well
    the model has recovered the true parameter values.

    Can be used with either point estimates or posterior samples.

    Args:
        true (np.ndarray): True parameter values, shape (n_observations, n_params).
        estimated (np.ndarray): Estimated parameter values. Can either be provided as
        a 2D array of shape shape (n_observations, n_params), or as a 3D array of shape
        (n_samples, n_observations, n_params), in which case the mean of the samples is
        plotted.
        param_names (List[str], optional): List of parameter names. Defaults to None.
        show_correlation (bool, optional): Whether to show the correlation coefficient
        in the title of the plot. Defaults to True.
        scale (float, optional): Scale of the plot. Defaults to 1.0.
        save_path (str, optional): Path to save the plot to. Defaults to None.
        save_fname (str, optional): File name for the saved plot, if `save_path` is not None.
        If None, the file name is generated automatically, including the current time and date.
        Defaults to None.
    """

    # Get mean of samples if provided
    if estimated.ndim == 3:
        estimated = estimated.mean(axis=0)
    
    # Plot
    f, ax = plt.subplots(1, true.shape[1], figsize=((2.333 * scale) * true.shape[1], 2.8 * scale))

    # Loop over parameters
    for i in range(true.shape[1]):

        # Plot values
        ax[i].scatter(true[:, i], estimated[:, i])

        # Axis labels
        if i == 0:
            ax[i].set_ylabel("Estimated")
        ax[i].set_xlabel("True")

        # Get the title of the plot
        if param_names is not None:
            title = param_names[i] + '\n' 
        else:
            title = ''

        # Add correlation coefficient to title
        if show_correlation:
            title += "r = {}".format(
                np.round(np.corrcoef(true[:, i], estimated[:, i])[0, 1], 2)
            )

        ax[i].set_title(title)

    plt.tight_layout()

    # Save plot if save_path is provided
    if save_path is not None:

        # Generate file name if not provided
        if save_fname is None:
            save_fname = 'recovery_plot_{}.svg'.format(
                datetime.now().strftime("%Y%m%d_%H%M%S")
            )

        plt.savefig(os.path.join(save_path, save_fname))


def plot_recovery_matrix(
    true: np.ndarray,
    estimated:np.ndarray,
    param_names:List[str]=None,
    scale:float=1.0,
    save_path:str=None,
    save_fname:str='recovery_matrix.svg'
):
    raise NotImplementedError("This function is not yet implemented")


def plot_pp(
    true: np.ndarray, 
    estimated:np.ndarray, 
    param_names:List[str]=None, 
    scale:float=1.0,
    save_path:str=None, 
    save_fname:str=None
):
    """
    Probability-probability plot. Plots the proportion of observations with values that fall within
    a given credible interval against the credible interval probability. Used for assessing how 
    well-calibrated the posterior is. Perfectly calibrated posteriors should result in points lying
    on the diagonal.

    NOTE: Designed for use with posterior distributions, cannot be used with point estimates.

    Args:
        true (np.ndarray): True parameter values, shape (n_observations, n_params).
        estimated (np.ndarray): Samples from posterior, shape (n_samples, n_observations, n_params).
        epoch (int): Epoch number.
        param_names (List[str], optional): List of parameter names. Defaults to None.
        save_path (str): Path to save the plot to.
        scale (float, optional): Scale of the plot. Defaults to 1.0.
        save_path (str, optional): Path to save the plot to. Defaults to None.
        save_fname (str, optional): File name for the saved plot, if `save_path` is not None.
        If None, the file name is generated automatically, including the current time and date.
        Defaults to None.
    """

    # Check number of dimensions in estimated values
    if estimated.ndim != 3:
        raise ValueError("Expected 3D array (n_samples, n_observations, n_params) for estimated values, got {}D".format(estimated.ndim))

    # Get number of params
    n_params = true.shape[1]

    # Plot
    f, ax = plt.subplots(1, true.shape[1], figsize=((2.333 * scale) * true.shape[1], 2.8 * scale))

    # Loop over parameters
    for param in range(n_params):
        
        # Axis labels
        if param == 0:
            ax[param].set_ylabel("Proportion of samples\nin CI")
        ax[param].set_xlabel("CI probability")

        # Get proportion of samples in CI
        ps = []
        
        n_samples = estimated.shape[0]

        # Iterate over observations
        for i in range(estimated.shape[1]):
            obs_samples = estimated[:, i, param]
            true_value = true[i, param]
            ps.append(np.sum(obs_samples > true_value) / float(n_samples))

        # Plot values
        ax[param].plot(np.linspace(0, 1, len(ps)), np.sort(ps))

        # Plot 45 degree line
        ax[param].plot([0, 1], [0, 1], color='black', linestyle='--')

        # Set title
        if param_names is not None:
            ax[param].set_title(param_names[param])

    plt.tight_layout()

    if save_path is not None:

        # Generate file name if not provided
        if save_fname is None:
            save_fname = 'pp_plot_{}.svg'.format(
                datetime.now().strftime("%Y%m%d_%H%M%S")
            )

        plt.savefig(os.path.join(save_path, save_fname))