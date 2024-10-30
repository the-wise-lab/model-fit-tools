import numpy as np


def expected_calibration_error(
    true: np.ndarray, estimated: np.ndarray, num_bins: int = 10
) -> np.ndarray:
    """
    Calculates Expected Calibration Error (ECE) for posterior distributions
    based on a p-p plot. ECE quantifies the average deviation of observed
    probabilities from expected probabilities across quantile bins.

    Args:
        true (np.ndarray): True parameter values, shape
            `(n_observations, n_params)`.
        estimated (np.ndarray): Samples from posterior, shape
            `(n_samples, n_observations, n_params)`.
        num_bins (int, optional): Number of bins for quantile
            calibration. Defaults to 10.

    Returns:
        np.ndarray: Expected Calibration Error for each parameter.
    """
    # Check dimensions
    if estimated.ndim != 3:
        raise ValueError(
            "Expected 3D array (n_samples, n_observations, n_params)"
            " for estimated values, got {}D".format(estimated.ndim)
        )

    n_samples, n_observations, n_params = estimated.shape
    ece = np.zeros(n_params)

    # Loop over parameters
    for param in range(n_params):
        ps = []  # Collect proportion of samples in CI for each observation

        # Iterate over observations
        for i in range(n_observations):
            obs_samples = estimated[:, i, param]
            true_value = true[i, param]
            ps.append(np.sum(obs_samples > true_value) / float(n_samples))

        # Bin boundaries for calibration
        bin_edges = np.linspace(0, 1, num_bins + 1)

        # Compute ECE for this parameter
        ece_param = 0.0
        for j in range(num_bins):
            in_bin = (np.array(ps) >= bin_edges[j]) & (
                np.array(ps) < bin_edges[j + 1]
            )
            if np.any(in_bin):
                observed_prob = np.mean(np.array(ps)[in_bin])
                expected_prob = (bin_edges[j] + bin_edges[j + 1]) / 2
                ece_param += np.abs(observed_prob - expected_prob) * np.mean(
                    in_bin
                )

        ece[param] = ece_param

    return ece
