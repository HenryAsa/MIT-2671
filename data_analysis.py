import matplotlib.pyplot as plt
import librosa
import numpy as np
import os
import pandas as pd
from scipy.io import wavfile
from scipy.spatial.distance import euclidean
from scipy.optimize import curve_fit
from pydub import AudioSegment

from audio_file_rep import AudioFile
from audio_modifications import average_audio_files, normalize_audio
from constants import DATA_AUDIO_SAMPLES_DIRECTORY, DATA_DIRECTORY, DATA_NORMALIZED_SAMPLES_DIRECTORY, DATA_RECORDED_SAMPLES_DIRECTORY
from utils import get_filetype_from_folder, remove_duplicates_from_list, compute_confidence_interval


def load_data_by_paritions(
        time_folder_name: str,
        generate_normalized_files: bool = True,
    ) -> dict[str, dict[str, list[AudioFile]]]:
    """
    Load and organize audio data by partitions from specified time
    folders.

    This function scans a designated directory structure for audio
    files, organizing them into a structured dictionary based on the
    song folder, sample rate, and file type. It identifies a master
    sample for each song and separates files by their sample rate and
    file type, facilitating easier access to specific types of audio
    data for processing or analysis.

    Parameters
    ----------
    time_folder_name : str
        The name of the folder within the data directory from which to
        load audio samples. This folder is expected to contain
        subfolders for recorded samples and possibly for audio
        samples, each containing audio files in '.wav' or '.mp3'
        format.

    Returns
    -------
    dict[str, dict[str, list[AudioFile]]]
        A dictionary where each key is the name of a song folder, and
        the value is another dictionary with keys 'master_filepath',
        'by_sample_rate', and 'by_file_type'. 'master_filepath' is the
        path to the master audio file for that song. 'by_sample_rate'
        and 'by_file_type' are dictionaries that organize `AudioFile`
        objects by their sample rates and file types, respectively.

    Examples
    --------
    >>> load_data_by_partitions('2023-March')
    {'song1': {'master_filepath': 'path/to/song1_master_S192000_B24.wav',
               'by_sample_rate': {192000: [<AudioFile object>, ...]},
               'by_file_type': {'wav': [<AudioFile object>, ...]}},
     ...}

    Notes
    -----
    - The `AudioFile` class is assumed to have attributes `file_path`,
      `file_type`, and `sample_rate`, and to be initialized with a
      file path.
    - This function does not return audio samples from the
      'audio_samples_folder_path' due to the commented-out line, which
      may be an oversight or intentional.
    - The master sample file for each song is identified by a specific
      naming convention: a file ending with "_S192000_B24.wav".
    """
    output: dict[str, dict] = {}

    recorded_samples_folder_path = f'{DATA_DIRECTORY}/{time_folder_name}/{DATA_RECORDED_SAMPLES_DIRECTORY}'
    audio_samples_folder_path = f'{DATA_DIRECTORY}/{time_folder_name}/{DATA_AUDIO_SAMPLES_DIRECTORY}'
    normalized_samples_folder_path = f'{DATA_DIRECTORY}/{time_folder_name}/{DATA_NORMALIZED_SAMPLES_DIRECTORY}'

    for song_folder in os.listdir(recorded_samples_folder_path):
        recorded_filepaths = set(get_filetype_from_folder(f'{recorded_samples_folder_path}/{song_folder}', '.wav')).union(set(get_filetype_from_folder(f'{recorded_samples_folder_path}/{song_folder}', '.mp3')))
        audio_samples_filepaths = set(get_filetype_from_folder(f'{audio_samples_folder_path}/{song_folder}', '.wav')).union(set(get_filetype_from_folder(f'{audio_samples_folder_path}/{song_folder}', '.mp3')))

        if len(recorded_filepaths) == 0:
            print(f'SKIPPING THE SONG FOLDER "{song_folder}" BECAUSE IT DOESN\'T CONTAIN ANY RECORDINGS')
            continue

        by_sample_rate: dict[int, set[AudioFile]] = {}
        by_file_type: dict[str, set[AudioFile]] = {}

        #### NORMALIZE AUDIO FILES ####
        if generate_normalized_files:
            unique_files = set([audio_file[:audio_file.rfind("_")] for audio_file in recorded_filepaths])

            for unique_file in unique_files:
                all_trial_files = [audio_file for audio_file in recorded_filepaths if audio_file.startswith(unique_file)]
                average_audio_files(all_trial_files)
                normalize_audio(all_trial_files + [f'{unique_file}_AVG.wav'])
        ###############################

        normalized_filepaths = get_filetype_from_folder(f'{normalized_samples_folder_path}/{song_folder}', '.wav')

        try:
            master_sample_path = [audio_file for audio_file in normalized_filepaths if audio_file.endswith("_S192000_B24_NORMALIZED.wav")][0]
        except:
            master_sample_path = [audio_file for audio_file in normalized_filepaths if audio_file.endswith("_S192000_B24_AVG_NORMALIZED.wav")][0]
            # master_samples_regex = re.compile(r'.*_S192000_B24_TRIAL\d+_NORMALIZED\.wav')
            # master_samples = [AudioFile(audio_file) for audio_file in recorded_filepaths if re.match(master_samples_regex, audio_file)]
            # average_audio_files([audio_file for audio_file in recorded_filepaths if re.match(master_samples_regex, audio_file)])
            # print(master_samples)
            # true_master = [AudioFile(audio_file) for audio_file in recorded_filepaths if audio_file.endswith("_S192000_B24_AVG.wav")]
            # normalize_audio([audio_file for audio_file in recorded_filepaths if re.match(master_samples_regex, audio_file)] + [true_master[0].file_path])
            
            # master_samples_regex_normalized = re.compile(r'.*_S192000_B24_TRIAL\d+_NORMALIZED\.wav')
            # master_samples = [AudioFile(audio_file) for audio_file in recorded_filepaths if re.match(master_samples_regex_normalized, audio_file)]
            # true_master = [AudioFile(audio_file) for audio_file in recorded_filepaths if audio_file.endswith("_S192000_B24_AVG_NORMALIZED.wav")]
            # compare_audio_samples_mse(true_master[0], master_samples, True)
            # master_sample_path = true_master[0].file_path

        # normalized_filepaths.discard(master_sample_path)

        for filename in sorted(normalized_filepaths):
            current_file = AudioFile(file_path=filename)

            if current_file.file_type not in by_file_type:
                by_file_type[current_file.file_type] = set()
            by_file_type[current_file.file_type].add(current_file)

            if current_file.sample_rate not in by_sample_rate:
                by_sample_rate[current_file.sample_rate] = set()
            by_sample_rate[current_file.sample_rate].add(current_file)

        output[song_folder] = {
            "master_filepath": master_sample_path,
            "by_sample_rate":  by_sample_rate,
            "by_file_type":    by_file_type,
        }

    return output


def compare_audio_samples_mse(
        master_sample_path: AudioFile,
        audio_samples: list[AudioFile],
        allow_resampling_when_mismatched: bool = False,
    ) -> None:
    """
    Compare the master audio sample with a list of audio samples using
    the Mean Squared Error (MSE) metric and visualize the comparison.

    This function loads the master audio sample and the list of audio
    samples provided, ensuring they are all at the same sample rate.
    It then calculates the MSE between the master sample and each of
    the audio samples, plots these samples for visual comparison, and
    prints the MSE values.

    Parameters
    ----------
    master_sample_path : AudioFile
        The audio file object representing the master sample.
    audio_samples : list[AudioFile]
        A list of audio file objects to compare against the master
        sample.

    Raises
    ------
    ValueError
        If there is a sample rate mismatch between the master sample
        and any of the audio samples.

    Notes
    -----
    - The function assumes that all audio files are in a format that
      scipy's wavfile module can read.
    - Audio samples with different lengths or bit depths are handled
      by converting to float32 and trimming to the minimum length.
    - This function is designed for visual and quantitative comparison
      but does not return any values.

    Examples
    --------
    >>> master_sample = AudioFile('master_sample.wav')
    >>> samples = [AudioFile('sample1.wav'), AudioFile('sample2.wav')]
    >>> compare_audio_samples_mse(master_sample, samples)
    MSE for sample1.wav: 0.002
    MSE for sample2.wav: 0.003
    """
    # Load the master sample
    sr_master, master_sample = wavfile.read(master_sample_path.file_path)

    # Convert to float for consistency in calculation, especially if
    # the files have different bit depths
    master_sample = master_sample.astype(np.float32)

    # Initialize a plot
    plt.figure(figsize=(10, 6))
    times_master = np.arange(master_sample.size) / sr_master
    plt.plot(times_master, master_sample, label='Master Sample', alpha=0.5)

    mse_values = []

    for audio_file in audio_samples:
        # Load the comparison sample
        sr_sample, sample = wavfile.read(audio_file.file_path)
        # Convert to float to match master sample
        sample = sample.astype(np.float32)

        # Ensure the sampling rates match, resample if necessary (simple approach)
        if sr_sample != sr_master:
            if allow_resampling_when_mismatched:
                print(f'RESAMPLED - Sample Rate Mismatch: Master({sr_master}) vs Sample({sr_sample}) in {audio_file}')
                sample = librosa.resample(sample, orig_sr=sr_sample, target_sr=sr_master)
                sr_sample = master_sample
            else:
                raise ValueError(f'Sample rate mismatch: Master({sr_master}) vs Sample({sr_sample}) in {audio_file}')

        # Ensure the samples have the same length
        min_len = min(master_sample.size, sample.size)
        master_sample_trimmed = master_sample[:min_len]
        sample_trimmed = sample[:min_len]

        # Calculate MSE
        mse = np.mean((master_sample_trimmed - sample_trimmed) ** 2)
        mse_values.append(mse)

        # Plot comparison sample
        times_sample = np.arange(sample_trimmed.size) / sr_sample
        plt.plot(
            times_sample,
            sample_trimmed,
            label=f'{audio_file.get_by_sample_rate_name()}',
            alpha=0.4
        )

    # Finalize plot
    plt.title('Comparison of Audio Samples')
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    # plt.show()

    # Print MSE values
    for path, mse in zip(audio_samples, mse_values):
        print(f'MSE for {path}: {mse}')


def plot_error_from_master(
        master_sample_path: AudioFile,
        audio_samples: list[AudioFile],
    ) -> None:
    """
    Plots the error signal derived from comparing a master audio
    sample with a list of other audio samples.

    This function first loads the master audio sample and each sample
    from the given list, converting them to float32 for precision. It
    then calculates the error signal for each sample as the difference
    between the master sample and the sample. The mean squared error
    (MSE) of the error signal is calculated and plotted for each
    sample. Finally, the MSE values are printed.

    Parameters
    ----------
    master_sample_path : AudioFile
        The path to the master audio sample file.
    audio_samples : list[AudioFile]
        A list of AudioFile objects representing the audio samples to
        be compared against the master sample.

    Raises
    ------
    ValueError
        If there is a sample rate mismatch between the master sample
        and any of the audio samples.

    Examples
    --------
    >>> master_path = AudioFile("master_sample.wav")
    >>> samples = [AudioFile("sample1.wav"), AudioFile("sample2.wav")]
    >>> plot_error_from_master(master_path, samples)
    This will plot the error signals and print the MSE values for the
    comparison of `master_sample.wav` with `sample1.wav` and
    `sample2.wav`.
    """
    # Load the master sample
    sr_master, master_sample = wavfile.read(master_sample_path.file_path)

    # Ensure master sample is in float format for precision
    master_sample = master_sample.astype(np.float32)

    mse_values = []

    # Set up the figure for plotting
    plt.figure(figsize=(14, len(audio_samples) * 2))

    for i, audio_file in enumerate(audio_samples, 1):
        # Load the comparison sample
        sr_sample, sample = wavfile.read(audio_file.file_path)
        sample = sample.astype(np.float32)

        if sr_sample != sr_master:
            raise ValueError(f"Sample rate mismatch: Master({sr_master}) vs Sample({sr_sample}) in {audio_file.file_path}")

        # Ensure the samples have the same length
        min_len = min(master_sample.size, sample.size)
        master_sample_trimmed = master_sample[:min_len]
        sample_trimmed = sample[:min_len]

        # Calculate the error (deviation) from the master sample
        error_signal = sample_trimmed - master_sample_trimmed

        # Calculate MSE for the error signal
        mse = np.mean(error_signal ** 2)
        mse_values.append(mse)

        # Plot the error signal
        plt.subplot(len(audio_samples), 1, i)
        times = np.arange(min_len) / sr_master
        plt.plot(times, error_signal, label=f'{audio_file.get_by_sample_rate_name()}')
        plt.title(f'Error from Master: {master_sample_path.file_path}')
        plt.xlabel('Time (s)')
        plt.ylabel('Error Amplitude')
        plt.legend()

    plt.tight_layout()
    # plt.show()

    # Print MSE values
    for path, mse in zip(audio_samples, mse_values):
        print(f'MSE for {path}: {mse}')


def load_audio_for_comparison(
        master_audio: AudioFile,
        audio_samples: list[AudioFile],
    ) -> tuple[list[float], list[float]]:
    """
    Load audio samples and compare them to a master audio file using
    Mean Squared Error (MSE).

    This function loads a master audio file and a list of audio
    samples, converts them to a consistent format, and calculates the
    Mean Squared Error (MSE) between each sample and the master audio
    file. It supports both WAV and MP3 formats. Samples are converted
    to float32 for consistency, and sample rates between the master
    and each sample must match. Currently, the function does not
    implement sample rate conversion and will raise an exception if a
    mismatch is detected.

    Parameters
    ----------
    master_audio : AudioFile
        The master audio file against which others are compared. Must
        have a `file_path` attribute.
    audio_samples : list[AudioFile]
        A list of audio files to compare against the master. Each must
        have a `file_path` attribute.

    Returns
    -------
    list
        A list of numpy arrays representing the error signals between
        the master audio sample and each of the audio samples.
    list
        A list of float values representing the mean squared error
        between the master audio sample and each of the audio samples.

    Raises
    ------
    ValueError
        If an audio file format is unsupported.
    NotImplementedError
        If sample rate conversion is required but not implemented.

    Examples
    --------
    >>> master = AudioFile('path/to/master.wav')
    >>> samples = [AudioFile('path/to/sample1.wav'), AudioFile('path/to/sample2.mp3')]
    >>> errors, mse_values = load_audio_for_comparison(master, samples)
    """
    # Load and convert the master sample to a consistent format
    sr_master, master_sample = wavfile.read(master_audio.file_path)
    master_sample = master_sample.astype(np.float32)

    errors = []
    mse_values = []

    for audio_file in audio_samples:
        path = audio_file.file_path
        # Determine the file type and load accordingly
        if path.endswith('.wav'):
            sr_sample, sample = wavfile.read(path)
            sample = sample.astype(np.float32)
        elif path.endswith('.mp3'):
            audio: AudioSegment = AudioSegment.from_mp3(path)
            # Convert to WAV-like data for consistency
            sample = np.array(audio.get_array_of_samples(), dtype=np.float32)
            sr_sample = audio.frame_rate
        else:
            raise ValueError(f"Unsupported file format for {path}")

        # Match the sample rate of the master sample
        if sr_sample != sr_master:
            raise NotImplementedError("Sample rate conversion is required but not implemented in this snippet.")
        
        # Ensure the samples have the same length
        min_len = min(master_sample.size, sample.size)
        master_sample_trimmed = master_sample[:min_len]
        sample_trimmed = sample[:min_len]

        # Calculate the error signal and MSE
        error_signal = sample_trimmed - master_sample_trimmed
        errors.append(error_signal)
        mse = np.mean(error_signal ** 2)
        mse_values.append(mse)

    return errors, mse_values


def audio_similarity_euclidean_mfcc(
        file1: str,
        file2: str,
        sr: int = 22050
    ) -> float:
    """
    Compare two audio files for similarity.

    This function computes the distance between the Mel-Frequency
    Cepstral Coefficients (MFCCs) of two audio files, which may vary
    in codec, sample rates, and bit depths.  A lower distance metric
    indicates greater similarity.

    Parameters
    ----------
    file1 : str
        Path to the first audio file.
    file2 : str
        Path to the second audio file.
    sr : int, optional
        Target sample rate for audio files. Default is 22050.

    Returns
    -------
    float
        Distance metric indicating the similarity between the audio
        files.

    Notes
    -----
    Both audio files are loaded and normalized to the same loudness
    before computing MFCCs. This ensures that the comparison focuses
    more on timbral textures rather than volume differences.

    Examples
    --------
    >>> audio_similarity("path/to/file1.wav", "path/to/file2.mp3")
    0.5
    """
    # Load and normalize audio files
    y1, _ = librosa.load(file1, sr=sr)
    y2, _ = librosa.load(file2, sr=sr)

    # Optional: Normalization step, if loudness is a concern
    # This can help ensure the comparison focuses more on the timbral texture rather than volume differences
    y1 = librosa.util.normalize(y1)
    y2 = librosa.util.normalize(y2)

    # Compute MFCCs from the normalized audio signals
    mfcc1 = librosa.feature.mfcc(y=y1, sr=sr)
    mfcc2 = librosa.feature.mfcc(y=y2, sr=sr)

    # Average the MFCCs across time
    avg_mfcc1 = np.mean(mfcc1, axis=1)
    avg_mfcc2 = np.mean(mfcc2, axis=1)

    # Calculate the Euclidean distance between the average MFCCs
    distance = euclidean(avg_mfcc1, avg_mfcc2)

    return distance


def plot_mfcc_error(
        file1: str,
        file2: str,
        sr: int=22050,
    ):
    """
    Plots the error of Mel Frequency Cepstral Coefficients (MFCCs)
    over time between two audio files.
    
    This function loads both files, normalizes their audio signals,
    computes the MFCCs for each, and then calculates and plots the
    absolute error between these MFCCs over time.

    Parameters
    ----------
    file1 : str
        Path to the first audio file.
    file2 : str
        Path to the second audio file.
    sr : int, optional
        Target sample rate for both audio files. Default is 22050.

    Returns
    -------
    None
        This function does not return a value. It generates and
        displays a plot of the MFCC error over time.

    Examples
    --------
    >>> plot_mfcc_error("audio1.wav", "audio2.wav", sr=22050)
        This will load the audio files `audio1.wav` and `audio2.wav`,
        compute their MFCCs at a sample rate of 22050 Hz, calculate
        the absolute error between these MFCCs, and plot the result.

    Notes
    -----
    - This function requires `librosa` for audio processing and
      `matplotlib.pyplot` for plotting.
    - The audio files must be accessible at the specified paths and
      must be in a format readable by `librosa.load`.
    - It is assumed that the MFCCs of both audio files have the same
      shape. If this is not the case, additional handling will be
      required.
    - The plot generated by this function provides insights into how
      similar or different the audio content of the two files is in
      terms of MFCCs, which can be useful in applications such as
      audio analysis and comparison.
    """
    # Load and normalize audio files
    y1, _ = librosa.load(file1, sr=sr)
    y2, _ = librosa.load(file2, sr=sr)

    y1 = librosa.util.normalize(y1)
    y2 = librosa.util.normalize(y2)

    # Compute MFCCs from the normalized audio signals
    mfcc1 = librosa.feature.mfcc(y=y1, sr=sr)
    mfcc2 = librosa.feature.mfcc(y=y2, sr=sr)

    # Ensure the MFCCs have the same shape
    min_length = min(mfcc1.shape[1], mfcc2.shape[1])
    mfcc1 = mfcc1[:, :min_length]
    mfcc2 = mfcc2[:, :min_length]

    # Assuming both MFCC matrices have the same shape; if not, additional handling is needed
    error = np.abs(mfcc1 - mfcc2)  # Absolute error between MFCCs

    # Plot the error
    # plt.figure(figsize=(10, 4))
    # plt.imshow(error, aspect='auto', origin='lower', interpolation='nearest')
    # plt.title('MFCC Error Over Time')
    # plt.xlabel('Time')
    # plt.ylabel('MFCC Coefficients')
    # plt.colorbar(label='Absolute Error')
    # plt.tight_layout()
    # # plt.show()
    # plt.close()


def pad_mfcc(
        mfcc: np.ndarray,
        target_shape: tuple[int],
    ):
    """
    Pads the MFCC (Mel Frequency Cepstral Coefficients) array to
    match a specified shape by adding zeros if necessary.

    Parameters
    ----------
    mfcc : ndarray
        The MFCC array to pad, typically with a shape of 
        (n_mfcc, time_frames), where `n_mfcc` is the number of
        MFCC features and `time_frames` is the number of time frames.
    target_shape : tuple of int
        The target shape as a tuple, (n_mfcc, target_time_frames), 
        where `target_time_frames` is the desired number of time
        frames after padding.

    Returns
    -------
    ndarray
        The padded MFCC array with the shape specified by
        `target_shape`. If the original array already meets or exceeds
        the target in the second dimension (time_frames), it is
        returned unchanged.

    Examples
    --------
    >>> mfcc = np.array([[1, 2, 3], [4, 5, 6]])
    >>> target_shape = (2, 5)
    >>> pad_mfcc(mfcc, target_shape)
    array([[1, 2, 3, 0, 0],
           [4, 5, 6, 0, 0]])

    Note
    ----
    This function is useful for preparing MFCC features for input into
    models that require a fixed input shape, such as certain types of 
    neural networks.
    """
    padding_width = target_shape[1] - mfcc.shape[1]
    if padding_width > 0:
        # Pad the array if it's shorter than the target shape
        return np.pad(mfcc, ((0, 0), (0, padding_width)), mode='constant', constant_values=0)
    else:
        # Return the array unmodified if it's already the target length or longer
        return mfcc


def compare_files_to_master(
        master_file: AudioFile,
        file_group: list[AudioFile],
        sr: int = 22050
    ):
    """
    Compare audio files to a master audio file using MFCCs.

    This function loads a master audio file and a group of audio files,
    normalizes them, and computes their Mel-frequency cepstral
    coefficients (MFCCs). It then adjusts the MFCCs to have the same
    shape across all files by padding or truncating them. Finally, it
    calculates the absolute error between each file's MFCC and the
    master's MFCC, and visualizes these errors in a series of plots.

    Parameters
    ----------
    master_file : AudioFile
        The master audio file against which other files are compared.
    file_group : list[AudioFile]
        A list of audio files to be compared to the master file.
    sr : int, optional
        The sampling rate to use for loading audio files. Defaults to
        22050 Hz.

    Notes
    -----
    - The `AudioFile` class must have `file_path`, `song_simple_name`,
      `bit_depth`, and `sample_rate` attributes, and a
      `get_by_sample_rate_name` method.
    - This function requires `librosa` for audio processing and
      `matplotlib.pyplot` for plotting.

    Examples
    --------
    >>> master = AudioFile(file_path='master.wav', song_simple_name='Test Song',
                           bit_depth=16, sample_rate=44100)
    >>> files = [AudioFile(file_path=f'file{i}.wav', song_simple_name=f'File {i}',
                           bit_depth=16, sample_rate=44100) for i in range(1, 4)]
    >>> compare_files_to_master(master, files)
    This will load the master file and each file in `files`, compute
    their MFCCs, and plot the absolute error between each file's MFCCs
    and the master's MFCCs.
    """
    # Load and normalize the master audio file
    y_master, _ = librosa.load(master_file.file_path, sr=sr)
    y_master = librosa.util.normalize(y_master)
    mfcc_master = librosa.feature.mfcc(y=y_master, sr=sr)

    # Determine the maximum number of columns (time frames) across all MFCCs
    max_columns = mfcc_master.shape[1]
    for audio_file in file_group:
        y_file, _ = librosa.load(audio_file.file_path, sr=sr)
        y_file = librosa.util.normalize(y_file)
        mfcc_file = librosa.feature.mfcc(y=y_file, sr=sr)
        max_columns = max(max_columns, mfcc_file.shape[1])

    # Adjust mfcc_master to have the same shape as the longest MFCC array
    mfcc_master = pad_mfcc(mfcc_master, (mfcc_master.shape[0], max_columns))

    # Initialize error storage
    errors = []

    # Pre-compute errors
    for audio_file in file_group:
        y_file, _ = librosa.load(audio_file.file_path, sr=sr)
        y_file = librosa.util.normalize(y_file)
        mfcc_file = librosa.feature.mfcc(y=y_file, sr=sr)

        # Pad or truncate each MFCC array before computing the error
        mfcc_file_padded = pad_mfcc(mfcc_file, (mfcc_file.shape[0], max_columns))

        error = np.abs(mfcc_file_padded - mfcc_master)
        errors.append(error)

    # Determine global error bounds
    min_error = min(error.min() for error in errors)
    max_error = max(error.max() for error in errors)

    # Initialize a plot
    plt.figure(figsize=(10, len(file_group)))
    plt.suptitle(f'{master_file.song_simple_name} - Error Between Samples at {audio_file.sample_rate} Hz and a {master_file.bit_depth}-bit, {master_file.sample_rate} Hz Master')

    for i, (audio_file, error) in enumerate(zip(file_group, errors), start=1):
        plt.subplot(round(len(file_group) / 2), 2, i)
        plt.imshow(error, aspect='auto', origin='lower', cmap='hot', interpolation='nearest', vmin=min_error, vmax=max_error)
        plt.title(f'{audio_file.get_by_sample_rate_name()}')
        plt.xlabel('Time')
        plt.ylabel('MFCC Coefficients')
        plt.colorbar(label='Absolute Error')

    plt.tight_layout()
    # plt.show()
    # plt.close()


def generate_plots_background_report():
    time_folder = '04-02_00-44'
    audio_recording_data = load_data_by_paritions(time_folder_name=time_folder)

    for folder, folder_partitions in audio_recording_data.items():

        master_sample_path: str = folder_partitions["master_filepath"]
        by_sample_rate: dict[int, list[AudioFile]] = folder_partitions["by_sample_rate"]

        # Flatten the list of AudioFile objects across all dictionary values
        audio_files = sorted([file for files_list in by_sample_rate.values() for file in files_list])

        # Extract unique column names by calling get_by_sample_rate_name on each AudioFile
        # Use a set to avoid duplicate column names, if that's a concern
        column_names = remove_duplicates_from_list([audio_file.get_by_sample_rate_name() for audio_file in audio_files])

        # Create the DataFrame with these unique column names
        summary_data = pd.DataFrame(columns=column_names)

        for sample_rate, files in by_sample_rate.items():
            sample_paths = sorted(files)

            # plot_error_from_master(AudioFile(file_path=master_sample_path), sample_paths)     ## NOT VERY USEFUL
            # compare_audio_samples_mse(AudioFile(file_path=master_sample_path), sample_paths)

            # sample_paths.append(AudioFile(master_sample_path))
            # compare_files_to_master(AudioFile(master_sample_path), sample_paths)
            # plt.savefig(f'plots/{time_folder}/{folder}_S{sample_rate}.svg')
            # plt.close()

            for file in sample_paths:
                euclidean_distance = audio_similarity_euclidean_mfcc(master_sample_path, file.file_path)
                # print(euclidean_distance)
                summary_data.loc[file.sample_rate, file.get_by_sample_rate_name()] = euclidean_distance
                # plot_mfcc_error(master_sample_path, file.file_path)


        ##### PLOT DATAFRAME #####
        summary_data = summary_data.sort_index()

        for column in summary_data.columns:
            plt.plot(summary_data.index, summary_data[column], label=column)
        plt.title(f'{folder} MFCC Euclidean Distance By Sample Rate and Encoding')
        plt.xlabel('Sample Rate (Hz)')
        plt.ylabel('Euclidean Distance of the MFCC')
        plt.legend()
        # plt.show()
        plt.savefig(f'plots/{time_folder}/{folder}_distance.svg')
        plt.close()
        ##########################


def apply_curve_fit(dataframe: pd.DataFrame):
    def proportional(x, a):
        return a * x

    def linear(x, a, b):
        return a * x + b

    def quadratic(x, a, b, c):
        return a * x**2 + b * x + c

    def cubic(x, a, b, c, d):
        return a * x**3 + b * x**2 + c * x + d

    def exponential(x, a, b, c):
        return a * np.exp(b * x) + c

    def logarithmic(x, a, b, c):
        return a * np.log(b * x + 1) + c

    functions = [proportional, linear, quadratic, cubic, exponential, logarithmic]

    fig, ax = plt.subplots()
    colors = plt.cm.viridis(np.linspace(0, 1, len(dataframe.columns)))  # Generate colors

    current_iteration = 0

    for column in dataframe.columns:  # Assuming first column is index or non-numeric
        x_data = []
        y_data = []

        # Accumulate all y-values and their corresponding x-values
        for index, row in dataframe.iterrows():
            y_values = row[column]
            if isinstance(y_values, list):
                x_data.extend([index] * len(y_values))
                y_data.extend(y_values)

        if len(y_data) < 4:  # Need at least 4 data points to fit a cubic polynomial
            current_iteration += 1
            continue

        x_data = np.array(x_data)
        y_data = np.array(y_data)

        best_func = None
        best_rss = np.inf

        for func in functions:
            # Fit the curve
            try:
                popt, pcov = curve_fit(func, x_data, y_data, maxfev=10000)
                residuals = y_data - func(x_data[:len(y_data)], *popt)
                rss = np.sum(residuals**2)

                # if rss < best_rss:
                if func == logarithmic:
                    best_rss = rss
                    best_func = func
                    best_popt = popt

            except Exception as e:
                print(f"Error fitting data in column {column}: {e}")
                continue

        if best_func:
            # Generate a dense set of x-values for plotting the smooth curve
            x_dense = np.linspace(min(x_data), max(x_data), 1000)
            y_fit = best_func(x_dense, *best_popt)

            ax.plot(x_data, y_data, 'o', label=f'Raw {column}', color=colors[current_iteration])
            ax.plot(x_dense, y_fit, '-', label=f'Fit {column} ({best_func.__name__})', color=colors[current_iteration])

        current_iteration += 1
    ax.legend()
    # plt.show()
    
    # plt.show()


def plot_euclidean_distances(
        data_dict: dict[int, list[AudioFile]],
        time_folder: str,
        current_folder: str,
        master_sample_path: str,
        is_by_sample_rate: bool = True,
    ) -> pd.DataFrame:

    analysis_name = "euclidean-distance"

    # Flatten the list of AudioFile objects across all dictionary values
    audio_files = sorted([file for files_list in data_dict.values() for file in files_list])

    # Extract unique column names by calling get_by_sample_rate_name on each AudioFile
    # Use a set to avoid duplicate column names, if that's a concern
    column_names = remove_duplicates_from_list([audio_file.get_by_sample_rate_name() for audio_file in audio_files])

    # Create the DataFrame with these unique column names
    summary_data = pd.DataFrame(columns=column_names)

    os.makedirs(f'plots/{time_folder}/{current_folder}/{analysis_name}', exist_ok=True)

    name_function = AudioFile.get_by_sample_rate_name if is_by_sample_rate else AudioFile.get_by_file_type_name

    for sample_rate, files in data_dict.items():
        sample_paths = sorted(files)
        sample_avg_paths = sorted([audio_file for audio_file in sample_paths if "_AVG_" in audio_file.file_path])

        for file in sample_paths:
            euclidean_distance = audio_similarity_euclidean_mfcc(master_sample_path, file.file_path)
            print(f'{euclidean_distance}\t{file.file_path.split("/")[-1]}')

            try:
                current_value = summary_data.loc[file.sample_rate, name_function(file)]
            except KeyError:
                summary_data.loc[file.sample_rate, name_function(file)] = []
                current_value = summary_data.loc[file.sample_rate, name_function(file)]

            if not isinstance(current_value, list):
                summary_data.loc[file.sample_rate, name_function(file)] = []
                current_value = summary_data.loc[file.sample_rate, name_function(file)]
            current_value.append(euclidean_distance)
            # plot_mfcc_error(master_sample_path, file.file_path)

    ##### PLOT DATAFRAME #####
    summary_data = summary_data.sort_index()
    summary_data.to_csv(f'plots/{time_folder}/{current_folder}/{analysis_name}/{analysis_name}_summary.csv')

    apply_curve_fit(summary_data)

    plt.title(f'{current_folder} MFCC Euclidean Distance By Sample Rate and Encoding')
    plt.xlabel('Sample Rate (Hz)')
    plt.ylabel('Euclidean Distance of the MFCC')
    plt.legend()
    plt.savefig(f'plots/{time_folder}/{current_folder}/{analysis_name}/{analysis_name}_summary.svg')
    plt.savefig(f'plots/{time_folder}/{current_folder}/{analysis_name}/{analysis_name}_summary.png')
    plt.show()
    plt.close()
    ##########################

    return summary_data


        master_sample_path: str = folder_partitions["master_filepath"]
        by_sample_rate: dict[int, list[AudioFile]] = folder_partitions["by_sample_rate"]

        # Flatten the list of AudioFile objects across all dictionary values
        audio_files = sorted([file for files_list in by_sample_rate.values() for file in files_list])

        # Extract unique column names by calling get_by_sample_rate_name on each AudioFile
        # Use a set to avoid duplicate column names, if that's a concern
        column_names = remove_duplicates_from_list([audio_file.get_by_sample_rate_name() for audio_file in audio_files])

        # Create the DataFrame with these unique column names
        summary_data = pd.DataFrame(columns=column_names)

        os.makedirs(f'plots/{time_folder}', exist_ok=True)
        for sample_rate, files in by_sample_rate.items():
            sample_paths = sorted(files)
            sample_avg_paths = sorted([audio_file for audio_file in sample_paths if "_AVG_" in audio_file.file_path])

            # plot_error_from_master(AudioFile(file_path=master_sample_path), sample_paths)     ## NOT VERY USEFUL
            # compare_audio_samples_mse(AudioFile(file_path=master_sample_path), sample_paths)

            # # sample_paths.append(AudioFile(master_sample_path))
            # compare_files_to_master(AudioFile(master_sample_path), sample_paths)
            # plt.savefig(f'plots/{time_folder}/{folder}_S{sample_rate}.svg')
            # plt.close()

            # sample_avg_paths.append(AudioFile(master_sample_path))
            compare_files_to_master(AudioFile(master_sample_path), sample_avg_paths)
            plt.savefig(f'plots/{time_folder}/{folder}_S{sample_rate}.svg')
            plt.close()

            for file in sample_paths:
                euclidean_distance = audio_similarity_euclidean_mfcc(master_sample_path, file.file_path)
                print(f'{euclidean_distance}\t{file.file_path.split("/")[-1]}')

                try:
                    current_value = summary_data.loc[file.sample_rate, file.get_by_sample_rate_name()]
                except KeyError:
                    summary_data.loc[file.sample_rate, file.get_by_sample_rate_name()] = []
                    current_value = summary_data.loc[file.sample_rate, file.get_by_sample_rate_name()]

                if not isinstance(current_value, list):
                    summary_data.loc[file.sample_rate, file.get_by_sample_rate_name()] = []
                    current_value = summary_data.loc[file.sample_rate, file.get_by_sample_rate_name()]
                current_value.append(euclidean_distance)
                # plot_mfcc_error(master_sample_path, file.file_path)


    time_folder = '04-20_18-35'
    # time_folder = '04-21_17-53'
    audio_recording_data = load_data_by_paritions(time_folder_name=time_folder, generate_normalized_files=False)

    for folder, folder_partitions in audio_recording_data.items():

        master_sample_path: str = folder_partitions["master_filepath"]

        by_sample_rate: dict[int, list[AudioFile]] = folder_partitions["by_sample_rate"]
        #### PLOT EUCLIDEAN DISTANCE BY SAMPLE RATE ####
        summarized_euclidean_distance_by_sample_rate = plot_euclidean_distances(
            data_dict=by_sample_rate,
            time_folder=time_folder,
            current_folder=folder,
            master_sample_path=master_sample_path,
            is_by_sample_rate=True,
        )
        ################################################

        #### PLOT MEAN SQUARED ERROR BY SAMPLE RATE ####
        summarized_euclidean_distance_by_sample_rate = plot_mean_squared_error(
            data_dict=by_sample_rate,
            time_folder=time_folder,
            current_folder=folder,
            master_sample_path=master_sample_path,
            is_by_sample_rate=True,
        )
        ################################################

        by_file_type: dict[int, list[AudioFile]] = folder_partitions["by_file_type"]
        #### PLOT EUCLIDEAN DISTANCE BY FILE TYPE ####
        summarized_euclidean_distance_by_file_type = plot_euclidean_distances(
            data_dict=by_file_type,
            time_folder=time_folder,
            current_folder=folder,
            master_sample_path=master_sample_path,
            is_by_sample_rate=False,
        )
        ##############################################
