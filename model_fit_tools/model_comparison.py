import numpy as np
import pandas as pd


def calculate_confusion_and_inversion(
    df: pd.DataFrame, metric: str = "waic"
) -> (np.ndarray, np.ndarray):
    """
    Calculate confusion and inversion matrices based on WAIC values.

    This function computes the confusion matrix and inversion matrix normalized
    from a DataFrame containing simulation results. The confusion matrix
    reflects how often each model is selected as the best model when each model
    is true (i.e., `P(estimated | true)`). The inversion matrix normalized
    represents the proportion of times each model is estimated to be the best
    given each true model (i.e., `P(true | estimated)`).

    Args:
        df (pd.DataFrame): A DataFrame containing the following columns:\n
            - `"simulated_model"`: Label of the model used for simulation.
            - `"estimation_model"`: Label of the model used for estimation.
            - `"iteration"`: Iteration identifier (i.e., the iteration of
            the model comparison procedure, where multiple datasets are
            simulated and then fitted).\n
            A further column corresponding to the given metric should also be
            present. This column should correspond to the value of the
            `metric` argument.
        metric (str): The name of the column to be used as the metric for
        selecting
            the winning model. Higher values should indicate better models.
            Defaults to "waic".

    Returns:
        np.ndarray: The confusion matrix representing how often each model is
            estimated to be the best when each model is true.

        np.ndarray: The inversion matrix normalized, indicating the proportion
            of times each model is the true generating model given each
            estimated model.

    Example:
        `confusion_matrix, inversion_matrix =
        calculate_confusion_and_inversion(results)`

        Here, `results` is expected to be a DataFrame with at least the columns
        "simulated_model", "estimation_model", "iteration", and "waic".

        The results can then be plotted using
        `model_fit_tools.plotting.plot_matrices`:

        `model_fit_tools.plotting.plot_matrices(confusion_matrix,
        inversion_matrix, models)`


    """

    # Extract unique models and initialize matrix
    models = df["simulated_model"].unique()
    n_models = len(models)
    confusion_matrix = np.zeros((n_models, n_models))

    # Ensure the metric column exists in the dataframe
    if metric not in df.columns:
        raise ValueError(
            f"Metric {metric} is not a column in the provided DataFrame."
        )

    # Calculate confusion matrix
    for n, simulated_model in enumerate(models):
        for i in df["iteration"].unique():
            # Extract relevant run results
            run_results = df[
                (df["iteration"] == i)
                & (df["simulated_model"] == simulated_model)
            ].copy()

            # Convert the 'estimation_model' column to a Categorical type with
            # custom order
            run_results["estimation_model"] = pd.Categorical(
                run_results["estimation_model"],
                categories=models,
                ordered=True,
            )

            # Sort the DataFrame by 'estimation_model'
            run_results = run_results.sort_values("estimation_model")

            # Update confusion matrix
            confusion_matrix[np.argmax(run_results[metric].values), n] += 1

    # Normalize confusion matrix
    confusion_matrix = confusion_matrix / len(df["iteration"].unique())

    # Calculate inversion matrix
    inversion_matrix = confusion_matrix.T

    return confusion_matrix, inversion_matrix
