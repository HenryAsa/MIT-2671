
from pathlib import Path
from datetime import datetime
import os
import numpy as np

from constants import DATA_AUDIO_SAMPLES_DIRECTORY, DATA_DIRECTORY, DATA_RECORDED_SAMPLES_DIRECTORY, RECORDED_SAMPLE_FILENAME_PREFIX


def initialize_data_folders() -> tuple[str, str]:
    initial_time = datetime.now().strftime("%m-%d_%H-%M")

    samples_output_directory = f'{DATA_DIRECTORY}/{initial_time}/{DATA_AUDIO_SAMPLES_DIRECTORY}'
    recorded_output_directory = f'{DATA_DIRECTORY}/{initial_time}/{DATA_RECORDED_SAMPLES_DIRECTORY}'

    os.makedirs(samples_output_directory, exist_ok=True)
    os.makedirs(recorded_output_directory, exist_ok=True)

    return samples_output_directory, recorded_output_directory


def get_audio_params_from_filepath(filepath: str) -> dict:
    print(filepath)
    file_params = str(filepath).split("/")[-1].split("_")
    output = {}
    output["filetype"] = f'.{file_params[-1].split(".")[-1]}'

    output["song_simple_name"] = f'{filepath.split("/")[-2]}'
    output["output_filename"] = f'{RECORDED_SAMPLE_FILENAME_PREFIX}{output["filetype"][1:]}_{filepath.split("/")[-1]}'
    output["sample_rate"] = int(file_params[-2][1:])

    if output["filetype"] == '.mp3':
        output["bitrate"] = int(file_params[-1].split(".")[0][2:-1])
    else:
        output["bit_depth"] = int(file_params[-1].split(".")[0][1:])

    return output


def get_files_from_folder(folder: str) -> list[str]:
    """
    Retrieve and sort file paths from a specified folder.

    This function lists all files in the given folder and returns
    their paths sorted alphabetically. Each file path is constructed
    by concatenating the folder path with the filename.

    Parameters
    ----------
    folder : str
        The path to the folder from which to retrieve file paths.

    Returns
    -------
    list[str]
        A list of sorted file paths present in the specified folder.

    Examples
    --------
    >>> get_files_from_folder('/my/folder')
    ['/my/folder/file1.txt', '/my/folder/file2.txt']
    """
    return sorted([f'{folder}/{filename}' for filename in os.listdir(folder)])


def get_filetype_from_folder(folder: str, filetype: str) -> list[str]:
    """
    Retrieve a sorted list of file paths in a specified folder.

    This function lists all files and directories in the given
    folder, combines them with the folder path to create full paths,
    and returns these paths in alphabetical order.  It does not
    recurse into subdirectories.

    Parameters
    ----------
    folder : str
        The path to the folder from which to list file paths.

    Returns
    -------
    list[str]
        A sorted list of file paths (including directories) in the
        specified folder.

    Examples
    --------
    >>> get_files_from_folder('/path/to/folder')
    ['/path/to/folder/dir1', '/path/to/folder/file1.txt']
    """
    return sorted(str(filepath) for filepath in Path(folder).rglob(f'*.{filetype[1:] if filetype.startswith(".") else filetype}'))


def map_to_discrete(
        array: np.ndarray,
        array_bounds: list[int],
        n_steps: int,
        n_bounds: list[int],
    ) -> np.ndarray:
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
    # # Unpack array bounds
    # min_val, max_val = array_bounds

    # # Compute the range of the values
    # value_range = max_val - min_val

    # # Unpack bounds for the output array
    # min_mapped_val, max_mapped_val = n_bounds

    # # Compute the step size for mapping
    # step = value_range / (n_steps - 1) if n_steps > 1 else 0

    # # Map each value to a discrete value
    # mapped_values = [(val - min_val) // step * step + min_mapped_val for val in array]

    # return np.array(mapped_values)

    desired_min, desired_max = n_bounds
    min_val, max_val = array_bounds

    normalized_array = (array - min_val) / (max_val - min_val)
    return normalized_array * (desired_max - desired_min) + desired_min
