

import numpy as np

def map_to_discrete(array: np.ndarray, array_bounds: list[int], n_steps: int, n_bounds: list[int]) -> np.ndarray:
    """
    Map values in the input array to discrete values within specified
    bounds

    Parameters
    ----------
    array : np.ndarray
        Input array of values
    array_bounds : list[int]
        Bounds for the input array, [min_val, max_val]
    n_steps : int
        Number of steps for mapping
    n_bounds : list[int]
        Bounds for the output array, [min_mapped_val, max_mapped_val]

    Returns
    -------
    np.ndarray
        Array of mapped values
    
    Notes
    -----
    The function maps values from the input array to discrete values
    within the specified bounds
    """
    # Unpack array bounds
    min_val, max_val = array_bounds

    # Compute the range of the values
    value_range = max_val - min_val

    # Unpack bounds for the output array
    min_mapped_val, max_mapped_val = n_bounds

    # Compute the step size for mapping
    step = value_range / (n_steps - 1) if n_steps > 1 else 0

    # Map each value to a discrete value
    mapped_values = [(val - min_val) // step * step + min_mapped_val for val in array]

    return np.array(mapped_values)
