"""Initial training data generation with Latin hypercube and Sobol' sampling.

The continuous sampling functions follow the data flow used by the original
``LHS-FMA.py`` and ``Sobol-FMA.py`` experiments: generate points in the unit
hypercube, scale them to the black-box variable bounds, and leave black-box
evaluation, discretization, and one-hot encoding to the caller.

Discrete and categorical variables are supported by a separate mapping
function.  Keeping this mapping separate makes the boundary between sampling
and encoding explicit while allowing mixed-variable tutorials to reuse the
same unit-hypercube designs.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import warnings

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import qmc


class SobolBalanceWarning(UserWarning):
    """Warning emitted when a Sobol' sample count is not a power of two."""





def _validate_sampling_shape(n_variables: int, n_samples: int) -> None:
    if isinstance(n_variables, bool) or not isinstance(n_variables, int):
        raise TypeError("n_variables must be an integer")
    if isinstance(n_samples, bool) or not isinstance(n_samples, int):
        raise TypeError("n_samples must be an integer")
    if n_variables <= 0:
        raise ValueError("n_variables must be positive")
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")


def _validate_unit_samples(samples: ArrayLike) -> NDArray[np.float64]:
    unit_samples = np.asarray(samples, dtype=float)
    if unit_samples.ndim != 2 or unit_samples.shape[1] == 0:
        raise ValueError("unit_samples must be a non-empty two-dimensional array")
    if not np.all(np.isfinite(unit_samples)):
        raise ValueError("unit_samples must contain only finite values")
    if np.any(unit_samples < 0.0) or np.any(unit_samples >= 1.0):
        raise ValueError("unit_samples must lie in the half-open interval [0, 1)")
    return unit_samples


def _validate_bounds(
    lower_bounds: ArrayLike,
    upper_bounds: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    lower = np.asarray(lower_bounds, dtype=float)
    upper = np.asarray(upper_bounds, dtype=float)
    if lower.ndim != 1 or upper.ndim != 1 or lower.size == 0:
        raise ValueError("lower_bounds and upper_bounds must be non-empty 1D arrays")
    if lower.shape != upper.shape:
        raise ValueError("lower_bounds and upper_bounds must have the same shape")
    if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
        raise ValueError("bounds must contain only finite values")
    if np.any(lower >= upper):
        raise ValueError("every lower bound must be smaller than its upper bound")
    return lower, upper


def _validate_candidates(
    variable_values: Sequence[Sequence[object]],
) -> tuple[NDArray[np.generic], ...]:
    candidates = tuple(np.asarray(values) for values in variable_values)
    if not candidates:
        raise ValueError("variable_values must contain at least one variable")
    if any(values.ndim != 1 or values.size == 0 for values in candidates):
        raise ValueError("each variable must have a non-empty 1D candidate list")
    return candidates


def lhs_unit_samples(
    n_variables: int,
    n_samples: int,
    *,
    seed: int | np.random.Generator | None = None,
) -> NDArray[np.float64]:
    """Generate Latin hypercube points in ``[0, 1)``."""
    _validate_sampling_shape(n_variables, n_samples)
    return qmc.LatinHypercube(d=n_variables, rng=seed).random(n=n_samples)


def sobol_unit_samples(
    n_variables: int,
    n_samples: int,
    *,
    seed: int | np.random.Generator | None = None,
    scramble: bool = True,
) -> NDArray[np.float64]:
    """Generate Sobol' points in ``[0, 1)``.

    Any positive sample count is accepted.  A power of two is recommended
    because that is the sample size for which Sobol' balance properties are
    guaranteed.  Non-power-of-two counts therefore emit a warning but still
    return the requested number of points.
    """
    _validate_sampling_shape(n_variables, n_samples)
    engine = qmc.Sobol(d=n_variables, scramble=scramble, rng=seed)
    if n_samples & (n_samples - 1) == 0:
        return engine.random_base2(m=n_samples.bit_length() - 1)
    warnings.warn(
        "Sobol' balance properties are guaranteed for n_samples = 2**m; "
        f"generating the requested n_samples={n_samples} with random(n).",
        SobolBalanceWarning,
        stacklevel=2,
    )
    return engine.random(n=n_samples)


def scale_continuous_samples(
    unit_samples: ArrayLike,
    lower_bounds: ArrayLike,
    upper_bounds: ArrayLike,
) -> NDArray[np.float64]:
    """Scale unit-hypercube samples to continuous black-box bounds."""
    samples = _validate_unit_samples(unit_samples)
    lower, upper = _validate_bounds(lower_bounds, upper_bounds)
    if samples.shape[1] != lower.size:
        raise ValueError("the sample dimension must match the number of bounds")
    return qmc.scale(samples, lower, upper)


def lhs_initial_samples(
    lower_bounds: ArrayLike,
    upper_bounds: ArrayLike,
    n_samples: int,
    *,
    seed: int | np.random.Generator | None = None,
) -> NDArray[np.float64]:
    """Generate scaled continuous initial data with Latin hypercube sampling."""
    lower, upper = _validate_bounds(lower_bounds, upper_bounds)
    unit_samples = lhs_unit_samples(lower.size, n_samples, seed=seed)
    return qmc.scale(unit_samples, lower, upper)


def sobol_initial_samples(
    lower_bounds: ArrayLike,
    upper_bounds: ArrayLike,
    n_samples: int,
    *,
    seed: int | np.random.Generator | None = None,
    scramble: bool = True,
) -> NDArray[np.float64]:
    """Generate scaled continuous initial data with Sobol' sampling."""
    lower, upper = _validate_bounds(lower_bounds, upper_bounds)
    unit_samples = sobol_unit_samples(
        lower.size,
        n_samples,
        seed=seed,
        scramble=scramble,
    )
    return qmc.scale(unit_samples, lower, upper)


def map_unit_samples_to_discrete(
    unit_samples: ArrayLike,
    variable_values: Sequence[Sequence[object]],
) -> NDArray[np.generic]:
    """Map unit-hypercube samples to discrete or categorical candidates.

    Each candidate receives an equal-width interval.  Numeric-only variables
    preserve a common NumPy numeric dtype; mixed numeric/string variables use
    an object array so numeric values are not silently converted to strings.
    """
    samples = _validate_unit_samples(unit_samples)
    candidates = _validate_candidates(variable_values)
    if samples.shape[1] != len(candidates):
        raise ValueError("the sample dimension must match len(variable_values)")

    numeric_only = all(values.dtype.kind in "biufc" for values in candidates)
    dtype: np.dtype[np.generic] | type[object]
    dtype = np.result_type(*(values.dtype for values in candidates)) if numeric_only else object
    mapped = np.empty(samples.shape, dtype=dtype)
    for column, values in enumerate(candidates):
        indices = np.minimum(
            (samples[:, column] * values.size).astype(int),
            values.size - 1,
        )
        mapped[:, column] = values[indices]
    return mapped











def generate(
    method: str,
    n_samples: int,
    *,
    lower_bounds: ArrayLike | None = None,
    upper_bounds: ArrayLike | None = None,
    variable_values: Sequence[Sequence[object]] | None = None,
    seed: int | np.random.Generator | None = None,
    scramble: bool = True,
) -> NDArray[np.generic]:
    """Generate initial samples using LHS or Sobol' sequences.
    
    If lower_bounds and upper_bounds are provided, it generates continuous samples.
    If variable_values is provided, it generates discrete/categorical mapped samples.
    """
    method = method.lower()
    if method not in {"lhs", "sobol"}:
        raise ValueError("method must be 'lhs' or 'sobol'")
        
    if variable_values is not None:
        n_variables = len(variable_values)
        if method == "lhs":
            unit_samples = lhs_unit_samples(n_variables, n_samples, seed=seed)
        else:
            unit_samples = sobol_unit_samples(n_variables, n_samples, seed=seed, scramble=scramble)
            
        mapped = map_unit_samples_to_discrete(unit_samples, variable_values)
        
        # Check marginal coverage
        m = len(variable_values[0])
        if n_samples % m != 0:
            warnings.warn(
                "初期データ数と分割数が合っていないため，完全な網羅性は保証されません．",
                UserWarning,
                stacklevel=2,
            )
            
        return mapped
        
    elif lower_bounds is not None and upper_bounds is not None:
        if method == "lhs":
            return lhs_initial_samples(lower_bounds, upper_bounds, n_samples, seed=seed)
        else:
            return sobol_initial_samples(lower_bounds, upper_bounds, n_samples, seed=seed, scramble=scramble)
            
    else:
        raise ValueError("Either (lower_bounds, upper_bounds) or variable_values must be provided")


__all__ = [
    "SobolBalanceWarning",
    "generate",
    "lhs_initial_samples",
    "lhs_unit_samples",
    "map_unit_samples_to_discrete",
    "scale_continuous_samples",
    "sobol_initial_samples",
    "sobol_unit_samples",
]
