"""This module contains utilities for pretty-printing scikit-learn objects."""

import numpy as np
from ..base import BaseEstimator


def _changed_params(estimator):
    """Return dict (param_name: value) of parameters that differ from default."""
    params = estimator.get_params(deep=False)
    filtered_params = {}
    
    if hasattr(estimator.__class__, '_get_param_names'):
        init_func = estimator.__class__.__init__
        init_params = estimator._get_param_names()
    else:
        return params
    
    # Get default parameter values from __init__ signature
    import inspect
    try:
        sig = inspect.signature(init_func)
        init_defaults = {}
        for param_name in init_params:
            if param_name in sig.parameters:
                param = sig.parameters[param_name]
                if param.default is not inspect.Parameter.empty:
                    init_defaults[param_name] = param.default
    except (ValueError, TypeError):
        # If we can't get the signature, return all params
        return params
    
    # Compare current params with defaults
    for key, value in params.items():
        if key in init_defaults:
            default_value = init_defaults[key]
            # Handle numpy arrays properly
            if isinstance(value, np.ndarray) or isinstance(default_value, np.ndarray):
                if not _arrays_equal(value, default_value):
                    filtered_params[key] = value
            else:
                # For non-array values, use regular comparison
                try:
                    if value != default_value:
                        filtered_params[key] = value
                except ValueError:
                    # If comparison fails (e.g., for complex objects), include the param
                    filtered_params[key] = value
        else:
            # If no default found, include the parameter
            filtered_params[key] = value
    
    return filtered_params


def _arrays_equal(a, b):
    """Check if two values are equal, handling numpy arrays properly."""
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return np.array_equal(a, b)
    elif isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return False
    else:
        try:
            return a == b
        except ValueError:
            return False


def _EstimatorPrettyPrinter(params, offset=0, printer=repr):
    """Pretty print the dictionary 'params' to a string."""
    # Check if we should print only changed parameters
    options = np.get_printoptions()
    print_changed_only = getattr(options, 'print_changed_only', False)
    
    if print_changed_only and hasattr(params, 'get_params'):
        # This is an estimator, get only changed params
        params_to_print = _changed_params(params)
    elif isinstance(params, dict):
        params_to_print = params
    else:
        # For non-dict objects, just use repr
        return printer(params)
    
    if not params_to_print:
        return ''
    
    # Format the parameters
    items = []
    for key, value in params_to_print.items():
        item_str = f'{key}={printer(value)}'
        items.append(item_str)
    
    return ', '.join(items)
