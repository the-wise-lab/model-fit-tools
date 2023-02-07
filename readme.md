# Model fit utilities

This repository contains useful functions for evaluating model fit. It's currently fairly limited, but will be expanded in the future.

## Installation

To install this package, clone the repository and run `pip install -e .` from the root directory. It can then be used as a regular Python package, e.g. `from model_fit_tools.plotting import plot_recovery`.

To install other dependencies, run `pip install -r requirements.txt`. 

## Examples

To plot parameter recovery, use the `plot_recovery` function. For example:

```python
from model_fit_tools.plotting import plot_recovery

# Estimate parameters using your favourite method
# This can be either point estimates or posterior samples
parameter_estimates = model_fit(data)

# Plot parameter recovery
plot_recovery(
    true_values,  # The true parameter values used to simulate data
    parameter_estimates,  # The estimated parameter values
    ['alpha', 'beta', 'gamma']  # The names of the parameters
)
```

To plot a probability-probability (PP) plot (to assess posterior calibration) use the `plot_pp` function. For example:

```python
from model_fit_tools.plotting import plot_pp

# Estimate parameters using your favourite method
# Must be in the form of posterior samples
parameter_estimates = model_fit(data)

# Plot PP plot
plot_pp(
    true_values,  # The true parameter values used to simulate data
    parameter_estimates,  # The estimated parameter values
    ['alpha', 'beta', 'gamma']  # The names of the parameters
)
```