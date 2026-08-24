"""
Numerical utilities for handling extremely small numbers in optical depth calculations.

When computing optical depths for CMB photons and PBHs, we often encounter
values like τ ~ 10⁻⁶⁰ to 10⁻¹⁰⁰ which underflow to zero in standard float64.

This module provides:
  1. Logarithmic computation of optical depths
  2. Safe computation of P_≥1 = 1 - exp(-τ) for tiny τ
  3. Utilities for tracking and reporting very small numbers
"""

import numpy as np
import warnings


class TinyNumber:
    """
    Representation of extremely small positive numbers using log10.
    
    Stores log10(x) instead of x to avoid underflow.
    Useful for optical depths τ ~ 10⁻⁶⁰ to 10⁻¹⁰⁰.
    
    Examples
    --------
    >>> tau = TinyNumber(log10_value=-60)  # τ = 10⁻⁶⁰
    >>> print(tau)
    10^-60.00
    >>> tau.value  # Returns 0.0 (underflows)
    0.0
    >>> tau.log10  # Returns -60.0 (exact)
    -60.0
    """
    
    def __init__(self, value=None, log10_value=None):
        """
        Initialize from either value or log10(value).
        
        Parameters
        ----------
        value : float, optional
            The actual value (will underflow if too small)
        log10_value : float, optional
            log10 of the value (preferred for tiny numbers)
        """
        if log10_value is not None:
            self.log10 = float(log10_value)
        elif value is not None:
            if value > 0:
                self.log10 = np.log10(value)
            elif value == 0:
                self.log10 = -np.inf
            else:
                raise ValueError("TinyNumber requires positive value")
        else:
            raise ValueError("Must provide either value or log10_value")
    
    @property
    def value(self):
        """Get actual value (may underflow to 0)."""
        if np.isfinite(self.log10):
            return 10.0 ** self.log10
        else:
            return 0.0
    
    def __repr__(self):
        if np.isfinite(self.log10):
            return f"TinyNumber(10^{self.log10:.2f})"
        else:
            return f"TinyNumber(0)"
    
    def __str__(self):
        if np.isfinite(self.log10):
            return f"10^{self.log10:.2f}"
        else:
            return "0"
    
    def __float__(self):
        return self.value
    
    def __add__(self, other):
        """Add two tiny numbers in log space."""
        if isinstance(other, TinyNumber):
            if not np.isfinite(self.log10):
                return other
            if not np.isfinite(other.log10):
                return self
            # log(a + b) = log(a) + log(1 + b/a) = log(a) + log(1 + 10^(log(b)-log(a)))
            max_log = max(self.log10, other.log10)
            min_log = min(self.log10, other.log10)
            if max_log - min_log > 20:  # One term dominates
                return TinyNumber(log10_value=max_log)
            else:
                sum_log = max_log + np.log10(1.0 + 10.0**(min_log - max_log))
                return TinyNumber(log10_value=sum_log)
        else:
            return self.value + other
    
    def __mul__(self, other):
        """Multiply two tiny numbers in log space."""
        if isinstance(other, TinyNumber):
            return TinyNumber(log10_value=self.log10 + other.log10)
        else:
            return TinyNumber(log10_value=self.log10 + np.log10(abs(other)))


def compute_optical_depth_safe(integrand_values, integration_variable, 
                               return_log=False, min_log10=-300):
    """
    Compute optical depth integral safely for extremely small values.
    
    Instead of computing τ = ∫ f(x) dx directly (which may underflow),
    we track whether the result is extremely small and return it in log space.
    
    Parameters
    ----------
    integrand_values : array_like
        Values of the integrand f(x)
    integration_variable : array_like
        Integration variable x
    return_log : bool, default=False
        If True, return log10(τ) instead of τ
    min_log10 : float, default=-300
        Minimum log10 value to track (below this, return -inf)
    
    Returns
    -------
    tau : float or TinyNumber
        Optical depth (or its log10 if return_log=True)
    """
    # Compute integral
    tau = np.trapezoid(integrand_values, integration_variable)
    
    if tau == 0.0:
        # Check if integrand is non-zero but result underflowed
        if np.any(integrand_values > 0):
            # Estimate log10(tau) from maximum integrand value
            max_integrand = np.max(integrand_values)
            delta_x = np.mean(np.diff(integration_variable))
            n_points = len(integration_variable)
            # Rough estimate: tau ~ max_integrand * delta_x * n_points
            log10_tau_est = np.log10(max_integrand) + np.log10(delta_x) + np.log10(n_points)
            
            if log10_tau_est < min_log10:
                if return_log:
                    return -np.inf
                else:
                    return TinyNumber(log10_value=-np.inf)
            else:
                if return_log:
                    return log10_tau_est
                else:
                    return TinyNumber(log10_value=log10_tau_est)
        else:
            if return_log:
                return -np.inf
            else:
                return TinyNumber(log10_value=-np.inf)
    
    elif tau > 0 and np.isfinite(tau):
        if return_log:
            return np.log10(tau)
        else:
            return TinyNumber(value=tau)
    else:
        if return_log:
            return -np.inf
        else:
            return TinyNumber(log10_value=-np.inf)


def probability_at_least_one_scatter(tau, use_log=False):
    """
    Compute P(≥1 scatter) = 1 - exp(-τ) safely for tiny τ.
    
    For τ ≪ 1: P ≈ τ - τ²/2 + τ³/6 - ...
    For τ ~ 1: P = 1 - exp(-τ)
    For τ ≫ 1: P ≈ 1
    
    Parameters
    ----------
    tau : float or TinyNumber
        Optical depth
    use_log : bool, default=False
        If True, return log10(P) for tiny P
    
    Returns
    -------
    p : float or TinyNumber
        Probability (or its log10 if use_log=True and P is tiny)
    """
    if isinstance(tau, TinyNumber):
        tau_val = tau.value
        tau_log = tau.log10
    else:
        tau_val = float(tau)
        tau_log = np.log10(tau_val) if tau_val > 0 else -np.inf
    
    if tau_val == 0.0 or not np.isfinite(tau_val):
        # τ = 0 or underflowed
        if use_log:
            return tau_log  # log10(P) ≈ log10(τ) for tiny τ
        else:
            return TinyNumber(log10_value=tau_log)
    
    elif tau_val < 1e-10:
        # Use Taylor expansion: P ≈ τ for tiny τ
        if use_log:
            return tau_log
        else:
            return TinyNumber(log10_value=tau_log)
    
    elif tau_val < 0.1:
        # Use np.expm1 for accuracy: P = -expm1(-τ) = 1 - exp(-τ)
        p = -np.expm1(-tau_val)
        if use_log:
            return np.log10(p) if p > 0 else -np.inf
        else:
            return p
    
    else:
        # Standard formula
        p = 1.0 - np.exp(-tau_val)
        if use_log:
            return np.log10(p) if p > 0 else -np.inf
        else:
            return p


def format_tiny_number(value, precision=2):
    """
    Format a very small number for display.
    
    Parameters
    ----------
    value : float or TinyNumber
        Number to format
    precision : int, default=2
        Number of decimal places for exponent
    
    Returns
    -------
    str
        Formatted string like "10^-60.00" or "1.23e-15"
    """
    if isinstance(value, TinyNumber):
        if np.isfinite(value.log10):
            return f"10^{value.log10:.{precision}f}"
        else:
            return "0"
    
    val = float(value)
    
    if val == 0.0:
        return "0"
    elif not np.isfinite(val):
        return str(val)
    elif abs(val) < 1e-100:
        # Extremely small, estimate from machine precision
        return f"<10^-300"
    elif abs(val) < 1e-15:
        log10_val = np.log10(abs(val))
        return f"10^{log10_val:.{precision}f}"
    else:
        return f"{val:.{precision}e}"


def safe_divide(numerator, denominator, default=0.0):
    """
    Safely divide two numbers, handling tiny values and zeros.
    
    Parameters
    ----------
    numerator : float or TinyNumber
        Numerator
    denominator : float or TinyNumber
        Denominator
    default : float, default=0.0
        Value to return if division is undefined
    
    Returns
    -------
    result : float or TinyNumber
        numerator / denominator, or default if undefined
    """
    if isinstance(numerator, TinyNumber) and isinstance(denominator, TinyNumber):
        if np.isfinite(denominator.log10) and denominator.log10 > -np.inf:
            return TinyNumber(log10_value=numerator.log10 - denominator.log10)
        else:
            return default
    
    num_val = float(numerator) if not isinstance(numerator, TinyNumber) else numerator.value
    den_val = float(denominator) if not isinstance(denominator, TinyNumber) else denominator.value
    
    if den_val == 0.0 or not np.isfinite(den_val):
        return default
    else:
        return num_val / den_val


def warn_if_underflow(value, name="value", threshold=-100):
    """
    Warn if a value has underflowed to zero but should be tracked.
    
    Parameters
    ----------
    value : float
        Value to check
    name : str
        Name of the value for warning message
    threshold : float
        log10 threshold below which to warn
    """
    if value == 0.0:
        warnings.warn(
            f"{name} underflowed to zero. Consider using TinyNumber or log-space "
            f"computation for values below 10^{threshold}.",
            RuntimeWarning
        )


# Convenience functions for common operations

def log10_sum(log10_values):
    """
    Compute log10(sum(10^x_i)) from log10 values.
    
    Uses log-sum-exp trick for numerical stability.
    """
    log10_values = np.asarray(log10_values)
    finite_vals = log10_values[np.isfinite(log10_values)]
    
    if len(finite_vals) == 0:
        return -np.inf
    
    max_val = np.max(finite_vals)
    if not np.isfinite(max_val):
        return -np.inf
    
    # log10(sum(10^x_i)) = max + log10(sum(10^(x_i - max)))
    shifted = finite_vals - max_val
    sum_shifted = np.sum(10.0 ** shifted)
    
    if sum_shifted > 0:
        return max_val + np.log10(sum_shifted)
    else:
        return -np.inf


def log10_mean(log10_values, weights=None):
    """
    Compute log10(mean(10^x_i)) from log10 values.
    
    Parameters
    ----------
    log10_values : array_like
        log10 of values
    weights : array_like, optional
        Weights for weighted mean
    
    Returns
    -------
    float
        log10 of the mean
    """
    if weights is None:
        return log10_sum(log10_values) - np.log10(len(log10_values))
    else:
        weights = np.asarray(weights)
        weighted_log10 = log10_values + np.log10(weights)
        return log10_sum(weighted_log10) - np.log10(np.sum(weights))
