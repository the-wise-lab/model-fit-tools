import datetime
import os
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def generate_placeholder_param_names(n_params: int) -> List[str]:
    """
    Utility function. Generates a list of placeholder
    parameter names.

    Args:
        n_params (int): Number of parameters.

    Returns:
        List[str]: List of parameter names.
    """
    return ["Parameter {}".format(i) for i in range(n_params)]


def plot_recovery(
    true: np.ndarray,
    estimated: np.ndarray,
    param_names: Union[List[str], None] = None,
    show_correlation: bool = True,
    scale: float = 1.0,
    save_path: Union[str, None] = None,
    save_fname: Union[str, None] = None,
    colour_by: Union[str, None] = None,
    scatter_kwargs: Dict[str, Any] = {},
) -> None:
    """
    Plots recovered parameter values against true ones. Used to determine how
    well the model has recovered the true parameter values.

    Can be used with either point estimates or posterior samples.

    Args:
        true (np.ndarray): True parameter values, shape `(n_observations,
            n_params)`.
        estimated (np.ndarray): Estimated parameter values. Can
            either be provided as a 2D array of shape shape `(n_observations,
            n_params)`, or as a 3D array of shape `(n_samples, n_observations,
            n_params)`, in which case the mean of the samples is plotted.
        param_names (List[str], optional): List of parameter names. Defaults to
            `None`.
        show_correlation (bool, optional): Whether to show the
            correlation coefficient in the title of the plot. Defaults to
            `True`.
        scale (float, optional): Scale of the plot. Defaults to `1.0`.
        save_path (str, optional): Path to save the plot to. Defaults to
            `None`.
        save_fname (str, optional): File name for the saved plot, if
            `save_path` is not `None`. If `None`, the file name is generated
            automatically, including the current time and date. Defaults to
            `None`.
        colour_by (str, optional): Parameter name to colour the points by.
            Defaults to `None`.
        scatter_kwargs (Dict[str, Any], optional): Optional keyword arguments
            for customizing scatter plot appearance. Defaults to `{}`.
    """

    # Get mean of samples if provided
    if estimated.ndim == 3:
        estimated = estimated.mean(axis=0)

    # Create placeholder param names if they are not given
    if param_names is None:
        param_names = generate_placeholder_param_names(true.shape[1])

    # Plot
    f, ax = plt.subplots(
        1,
        true.shape[1],
        figsize=((2.333 * scale) * true.shape[1], 2.8 * scale),
    )

    # Default scatter plot settings
    default_scatter_kwargs = {"alpha": 0.5, "s": 10, **scatter_kwargs}

    # Update with supplied scatter plot settings
    scatter_kwargs = {**default_scatter_kwargs, **scatter_kwargs}

    # Remove any "c" options if using colour_by
    if colour_by is not None:
        scatter_kwargs.pop("c", None)

    # Loop over parameters
    for i in range(true.shape[1]):
        # Plot values, optinally colouring points by the value of the parameter
        # specified by `colour_by`
        if colour_by is not None:
            ax[i].scatter(
                true[:, i],
                estimated[:, i],
                c=true[:, param_names.index(colour_by)],
                **scatter_kwargs,
            )
        else:
            ax[i].scatter(true[:, i], estimated[:, i], **scatter_kwargs)

        # Axis labels
        if i == 0:
            ax[i].set_ylabel("Estimated")
        ax[i].set_xlabel("True")

        # Get the title of the plot
        if param_names is not None:
            title = param_names[i] + "\n"
        else:
            title = ""

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
            save_fname = "recovery_plot_{}.svg".format(
                datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            )

        plt.savefig(os.path.join(save_path, save_fname))


def plot_recovery_matrix(
    true: np.ndarray,
    estimated: np.ndarray,
    param_names: Union[List[str], None] = None,
    scale: float = 1.0,
    colorbar_scale: float = 1.0,
    xtick_rotation: float = 0,
    cmap: str = "viridis",
    vmin: Union[float, None] = None,
    vmax: Union[float, None] = None,
    save_path: Union[str, None] = None,
    save_fname: Union[str, None] = None,
    ax: Union[plt.Axes, None] = None,
) -> None:
    """
    Plots a matrix of the correlation coefficients between true and recovered
    parameter values. Used to determine how well the model has recovered the
    true parameter values.

    Args:
        true (np.ndarray): True parameter values, shape `(n_observations,
            n_params)`.
        estimated (np.ndarray): Estimated parameter values. Can
            either be provided as a 2D array of shape shape `(n_observations,
            n_params)`, or as a 3D array of shape `(n_samples, n_observations,
            n_params)`, in which case the mean of the samples is plotted.
        param_names (List[str], optional): List of parameter names. Defaults to
            None.
        scale (float, optional): Scale of the plot. Defaults to `1.0`.
            colorbar_scale (float, optional): Scale for the colorbar. Defaults
            to `1.0`.
        xtick_rotation (float, optional): Degrees by which to rotate x
            tick labels. Defaults to `0`.
        cmap (str, optional): Colormap to use. Defaults to `"viridis"`.
        vmin (float, optional): Minimum value. Defaults to `None`.
        vmax (float, optional): Maximum value. Defaults to `None`.
        save_path (str, optional): Path to save the plot to. Defaults to
            `None`.
        save_fname (str, optional): File name for the saved plot, if
            `save_path` is not `None`. If `None`, the file name is generated
            automatically, including the current time and date. Defaults to
            `None`.
        ax (plt.Axes, optional): Axes object to plot on. Defaults to `None`.
    """

    # Get mean of samples if provided
    if estimated.ndim == 3:
        estimated = estimated.mean(axis=0)

    # Determine the number of parameters
    n_params = true.shape[1]

    # Create placeholder param names if they are not given
    if param_names is None:
        param_names = generate_placeholder_param_names(n_params)

    # Get correlation matrix
    recovery_corrs = np.corrcoef(
        true.T,
        estimated.T,
    )[n_params:, :n_params]

    # Round recovery correlations to 2 decimal places for cleaner visualization
    recovery_corrs = np.round(recovery_corrs, 2)

    # Create a new plot if no axis is provided
    if ax is None:
        f, ax = plt.subplots(figsize=(3 * scale, 3 * scale))

    # Plot the recovery correlations matrix as a heatmap
    sns.heatmap(
        recovery_corrs,
        annot=True,
        cmap=cmap,
        square=True,
        vmin=vmin,
        vmax=vmax,
        cbar_kws={"label": "$r$", "shrink": 0.8 * colorbar_scale},
        ax=ax,
    )

    # Label x and y axis ticks with parameter names, rotate x-axis labels if
    # specified
    plt.xticks(
        [i + 0.5 for i in range(n_params)],
        param_names,
        rotation=xtick_rotation,
    )
    plt.yticks(
        [i + 0.5 for i in range(n_params)],
        param_names,
        rotation=0,
        va="center",
    )

    # Label x and y axes
    plt.xlabel("True")
    plt.ylabel("Recovered")

    # Label x and y axes
    plt.tight_layout()

    # Save plot if save_path is provided
    if save_path is not None:
        # Generate file name if not provided
        if save_fname is None:
            save_fname = "recovery_matrix_{}.svg".format(
                datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            )

        plt.savefig(os.path.join(save_path, save_fname))


def plot_pp(
    true: np.ndarray,
    estimated: np.ndarray,
    param_names: Union[List[str], None] = None,
    scale: float = 1.0,
    save_path: Union[str, None] = None,
    save_fname: Union[str, None] = None,
) -> None:
    """
    Probability-probability plot. Plots the proportion of observations with
    values that fall within a given credible interval against the credible
    interval probability. Used for assessing how well-calibrated the posterior
    is. Perfectly calibrated posteriors should result in points lying on the
    diagonal.

    > NOTE: Designed for use with posterior distributions, cannot be used with
    point estimates.

    Args:
        true (np.ndarray): True parameter values, shape `(n_observations,
            n_params)`.
        estimated (np.ndarray): Samples from posterior, shape
            `(n_samples, n_observations, n_params)`.
        param_names (List[str], optional): List of parameter names.
            Defaults to `None`.
        save_path (str): Path to save the plot to. Defaults to `None`.
        scale (float, optional): Scale of the plot. Defaults to `1.0`.
        save_path (str, optional): Path to save the plot to. Defaults to None.
        save_fname (str, optional): File name for the saved plot,
            if `save_path` is not None. If None, the file name is generated
            automatically, including the current time and date.
            Defaults to `None`.
    """

    # Check number of dimensions in estimated values
    if estimated.ndim != 3:
        raise ValueError(
            "Expected 3D array (n_samples, n_observations, n_params)"
            "for estimated values, got {}D".format(estimated.ndim)
        )

    # Get number of params
    n_params = true.shape[1]

    # Create placeholder param names if they are not given
    if param_names is None:
        param_names = generate_placeholder_param_names(n_params)

    # Plot
    f, ax = plt.subplots(
        1,
        true.shape[1],
        figsize=((2.333 * scale) * true.shape[1], 2.8 * scale),
    )

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
        ax[param].plot([0, 1], [0, 1], color="black", linestyle="--")

        # Set title
        if param_names is not None:
            ax[param].set_title(param_names[param])

    plt.tight_layout()

    if save_path is not None:
        # Generate file name if not provided
        if save_fname is None:
            save_fname = "pp_plot_{}.svg".format(
                datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            )

        plt.savefig(os.path.join(save_path, save_fname))


def plot_parameter_dists(
    estimated: np.ndarray,
    param_names: Union[List[str], None] = None,
    scale: float = 1.0,
    save_path: Union[str, None] = None,
    save_fname: Union[str, None] = None,
) -> None:
    """
    Plot parameter estimate distributions as histograms.

    Can be used with either point estimates or posterior samples.

    Args:
        estimated (np.ndarray): Estimated parameter values. Can either be
            provided as a 2D array of shape shape
            `(n_observations, n_params)`, or as a 3D array of shape
            `(n_samples, n_observations, n_params)`, in which
            case the mean of the samples is plotted.
        param_names (List[str], optional): List of parameter names.
            Defaults to `None`.
        scale (float, optional): Scale of the figure. Defaults to `1.0`.
        save_path (str, optional): Path to save the figure to.
            Defaults to `None`.
        save_fname (str, optional): Filename to use when saving.
            Defaults to `None`.
    """

    # Get mean of samples if provided
    if estimated.ndim == 3:
        estimated = estimated.mean(axis=0)

    # Get number of params
    n_params = estimated.shape[1]

    # Create placeholder param names if they are not given
    if param_names is None:
        param_names = generate_placeholder_param_names(n_params)

    # Plot
    f, ax = plt.subplots(
        1,
        estimated.shape[1],
        figsize=((2.333 * scale) * estimated.shape[1], 2.8 * scale),
    )

    # Loop over parameters
    for i in range(estimated.shape[1]):
        # Plot values
        ax[i].hist(estimated[:, i])

        # Axis labels
        if i == 0:
            ax[i].set_ylabel("Count")
        ax[i].set_xlabel("Value")

        # Get the title of the plot
        if param_names is not None:
            title = param_names[i] + "\n"
        else:
            title = "Parameter {}\n".format(i + 1)

        ax[i].set_title(title)

    plt.tight_layout()

    # Save plot if save_path is provided
    if save_path is not None:
        # Generate file name if not provided
        if save_fname is None:
            save_fname = "recovery_plot_{}.svg".format(
                datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            )

        plt.savefig(os.path.join(save_path, save_fname))


def plot_waic(
    waic_data: Union[Dict[str, Any], pd.DataFrame],
    best_model_idx: int,
    bar_kwargs: Dict[str, Any] = {},
    ylim_kwargs: Dict[str, Any] = {},
    fig_kwargs: Dict[str, Any] = {},
    rotate_xticks: bool = False,
    ax: Optional[plt.Axes] = None,
    colours: Optional[List[str]] = None,
    model_rename_dict: Optional[Dict[str, str]] = None,
) -> None:
    """
    Creates a bar plot of WAIC values with standard error for given models.

    This function visualizes WAIC values along with their respective standard
    error for various models, highlighting the "best" model with a different
    color.

    Args:
        waic_data (Union[Dict[str, Any], pd.DataFrame]): A dictionary where
            keys are strings representing model names and values are objects
            with `elpd_waic` and `se` attributes, or a DataFrame with columns
            `"model"`, `"waic"`, and `"se"`.
        best_model_idx (int): Index of the model considered as the best, which
            will  be highlighted with a different color in the plot.
        bar_kwargs (Dict[str, Any]): Optional keyword arguments for customizing
            bar appearance.
        ylim_kwargs (Dict[str, Any]): Optional keyword arguments for
            customizing y-axis limits.
        fig_kwargs (Dict[str, Any]): Optional keyword arguments for
            customizing figure attributes.
        rotate_xticks (bool, optional): Whether to rotate x-axis tick labels by
            45 degrees. Defaults to `False`.
        ax (Optional[plt.Axes]): Matplotlib axis to plot on. If `None`,
            a new figure and axis are created. colours (Optional[List[str]]):
            List of colors to use for each bar. If None, default Matplotlib
            colors are used. Defaults to `None`
        model_rename_dict (Optional[Dict[str, str]]): Dictionary mapping model
            names to their display names. Defaults to `None`.

    """
    # Check if axis is provided, else create a new figure and axis
    if ax is None:
        fig_kwargs = {"figsize": (2.5, 2.5), **fig_kwargs}
        fig, ax = plt.subplots(**fig_kwargs)

    # Extract color palette from Matplotlib
    pal = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # Process waic_data depending on whether it's a dict or DataFrame
    if isinstance(waic_data, pd.DataFrame):
        waic_values = waic_data["waic"].tolist()
        se_values = waic_data["se"].tolist()
        model_names = waic_data["model"].tolist()
    else:  # if waic_data is a dictionary
        waic_values = [i.elpd_waic for i in waic_data.values()]
        se_values = [i.se for i in waic_data.values()]
        model_names = list(waic_data.keys())

    # Determine colors for each bar, highlighting the best
    # model with a different color
    if colours is None:
        colours = [pal[0]] * len(model_names)
        colours[best_model_idx] = pal[1]

    # Default bar plot settings
    default_bar_kwargs = {"capsize": 5, "color": colours, **bar_kwargs}

    # Plot the bar chart
    ax.bar(
        x=range(len(model_names)),
        height=waic_values,
        yerr=se_values,
        **default_bar_kwargs,
    )

    # Default y-axis limit settings
    default_ylim_kwargs = {
        "bottom": min(waic_values) - 500,
        "top": max(waic_values) + 500,
        **ylim_kwargs,
    }
    ax.set_ylim(**default_ylim_kwargs)

    # Format model names for display
    if model_rename_dict is not None:
        model_names = [model_rename_dict[i] for i in model_names]

    # Configure x-axis ticks and labels
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names)

    # Set axis labels
    ax.set_xlabel("Model")
    ax.set_ylabel("WAIC")

    # Rotate x-axis tick labels if specified
    if rotate_xticks:
        ax.tick_params(axis="x", rotation=45)


def plot_best_models(
    df: pd.DataFrame,
    metric: str = "waic",
    highest_best: bool = True,
    subjects_per_row: int = 40,
    fig_width_scale: float = 0.15,
    fig_height_scale: float = 0.15,
    marker_size: int = 20,
) -> None:
    """
    Generate a scatter plot grid showing the model with the highest/lowest
    model fit metric per subject.

    Parameters:
        df (pd.DataFrame): DataFrame containing the data. Expected columns are
            'subject', 'model', and an additional column containing the model
            fit metric.
        metric (str, optional): The model fit metric to use. Defaults to
            `'waic'`.
        highest_best (bool, optional): Whether the highest value of the
            metric is the best. Defaults to `True`.
        subjects_per_row (int, optional): The number of subjects per row in the
            grid. Defaults to `40`.
        fig_width_scale (float, optional): Scaling factor for figure width.
            Defaults to `0.15`.
        fig_height_scale (float, optional): Scaling factor for figure height.
            Defaults to `0.15`.
        marker_size (int, optional): Size of the scatter plot markers.
            Defaults to `20`.
    """
    # Find the model with the best metric for each subject
    if highest_best:
        idx_best_metric = df.groupby("subject")[metric].idxmax()
    else:
        idx_best_metric = df.groupby("subject")[metric].idxmin()
    best_models = df.loc[idx_best_metric]

    # Sort by model
    best_models = best_models.sort_values(by="model")

    # Calculate the number of rows needed
    num_rows = int(np.ceil(len(best_models) / subjects_per_row))

    # Generate grid positions
    x = np.tile(np.arange(subjects_per_row), num_rows)[: len(best_models)]
    y = np.repeat(np.arange(num_rows), subjects_per_row)[: len(best_models)]

    # Create the scatter plot
    plt.figure(
        figsize=(
            subjects_per_row * fig_width_scale + 2,
            num_rows * fig_height_scale,
        )
    )
    unique_models = best_models["model"].unique()

    # Get matplotlib default colour cycle
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # Create a scatter plot for each model type,
    # this will automatically generate
    # legend items if we add a label argument
    for i, model in enumerate(unique_models):
        mask = best_models["model"] == model
        plt.scatter(
            x[mask], y[mask], color=colors[i], s=marker_size, label=model
        )

    # Remove axes
    plt.axis("off")

    # Add a legend
    plt.legend(title="Model", bbox_to_anchor=(1, 1), loc="upper left")

    # Tidy up the plot
    plt.tight_layout(rect=[0, 0, 0.85, 1])


def plot_matrices(
    confusion_matrix: np.array,
    inversion_matrix: np.array,
    model_names: list,
    scale: float = 1.0,
    cmap: str = "viridis",
) -> None:
    """
    Plot confusion and inversion matrices as heatmaps.

    Args:
        confusion_matrix (np.array): The confusion matrix, indicating how often
            each model is estimated to be the best when each model is true.
        inversion_matrix (np.array): The inversion matrix normalized,
            indicating the proportion of times each model is selected as best
            given each true model.
        model_names (list): List of strings indicating the name of each model,
            used for axis labels. scale (float): Scaling factor for the
            matrices.
        cmap (str): The colormap to use for the heatmaps.

    Returns:
        None: Plots the matrices as heatmaps.

    Example:
        plot_matrices(confusion_matrix, inversion_matrix, ['Model1', 'Model2',
        'Model3'])
    """

    fig, axs = plt.subplots(1, 2, figsize=(14 * scale, 6 * scale))

    # Replace underscores with spaces
    model_names = [name.replace("_", " ") for name in model_names]

    # Capitalize first letter
    model_names = [name.capitalize() for name in model_names]

    # Replace 'Mf' with 'MF' and 'Mb' with 'MB'
    model_names = [name.replace("Mf", "MF") for name in model_names]
    model_names = [name.replace("Mb", "MB") for name in model_names]

    # Plotting the confusion matrix
    sns.heatmap(
        data=confusion_matrix,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        square=True,
        xticklabels=model_names,
        yticklabels=model_names,
        ax=axs[0],
    )
    axs[0].set_title("Confusion Matrix")
    axs[0].set_xlabel("Simulated Model")
    axs[0].set_ylabel("Estimated Model")
    # set X tick labels at 45 degree angle
    axs[0].set_xticklabels(
        axs[0].get_xticklabels(), rotation=45, horizontalalignment="right"
    )

    # Plotting the inversion matrix
    sns.heatmap(
        data=inversion_matrix,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        square=True,
        xticklabels=model_names,
        yticklabels=model_names,
        ax=axs[1],
    )
    axs[1].set_title("Inversion Matrix")
    axs[1].set_xlabel("Simulated Model")
    axs[1].set_ylabel("Estimated Model")
    # set X tick labels at 45 degree angle
    axs[1].set_xticklabels(
        axs[1].get_xticklabels(), rotation=45, horizontalalignment="right"
    )

    # Adjust layout
    plt.tight_layout()
