import librosa
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from scipy.io import wavfile
from scipy.spatial.distance import euclidean
from pydub import AudioSegment

from audio_file_rep import AudioFile
from constants import DATA_AUDIO_SAMPLES_DIRECTORY, DATA_DIRECTORY, DATA_RECORDED_SAMPLES_DIRECTORY
from utils import get_audio_params_from_filepath, get_filetype_from_folder, remove_duplicates_from_list


def load_data_by_paritions(
        time_folder_name: str,
    ) -> dict[str, dict[str, list[AudioFile]]]:

    output: dict[str, dict] = {}

    recorded_samples_folder_path = f'{DATA_DIRECTORY}/{time_folder_name}/{DATA_RECORDED_SAMPLES_DIRECTORY}'
    audio_samples_folder_path = f'{DATA_DIRECTORY}/{time_folder_name}/{DATA_AUDIO_SAMPLES_DIRECTORY}'

    for folder in os.listdir(recorded_samples_folder_path):
        recorded_filepaths = set(get_filetype_from_folder(f'{recorded_samples_folder_path}/{folder}', '.wav')).union(set(get_filetype_from_folder(f'{recorded_samples_folder_path}/{folder}', '.mp3')))
        # audio_samples_filepaths = set(get_filetype_from_folder(f'{audio_samples_folder_path}/{folder}', '.wav')).union(set(get_filetype_from_folder(f'{audio_samples_folder_path}/{folder}', '.mp3')))

        by_sample_rate: dict[int, set[AudioFile]] = {}
        by_file_type: dict[str, set[AudioFile]] = {}

        master_sample_path = next((audio_file for audio_file in recorded_filepaths if audio_file.endswith("_S192000_B24.wav")))
        recorded_filepaths.discard(master_sample_path)

        for filename in sorted(recorded_filepaths):
            current_file = AudioFile(file_path=filename)

            if current_file.file_type not in by_file_type:
                by_file_type[current_file.file_type] = set()
            by_file_type[current_file.file_type].add(current_file)

            if current_file.sample_rate not in by_sample_rate:
                by_sample_rate[current_file.sample_rate] = set()
            by_sample_rate[current_file.sample_rate].add(current_file)

        output[folder] = {"master_filepath": master_sample_path, "by_sample_rate": by_sample_rate, "by_file_type": by_file_type}

    return output


def compare_audio_samples(
        master_sample_path: AudioFile,
        sample_paths: list[AudioFile]
    ) -> None:
    # Load the master sample
    sr_master, master_sample = wavfile.read(master_sample_path.file_path)

    # Convert to float for consistency in calculation, especially if the files have different bit depths
    master_sample = master_sample.astype(np.float32)

    # Initialize a plot
    plt.figure(figsize=(10, 6))
    times_master = np.arange(master_sample.size) / sr_master
    plt.plot(times_master, master_sample, label='Master Sample', alpha=0.5)

    mse_values = []

    for audio_file in sample_paths:
        # Load the comparison sample
        sr_sample, sample = wavfile.read(audio_file.file_path)

        # Ensure the sampling rates match, resample if necessary (simple approach)
        if sr_sample != sr_master:
            raise ValueError(f"Sample rate mismatch: Master({sr_master}) vs Sample({sr_sample}) in {audio_file}")

        # Convert to float to match master sample
        sample = sample.astype(np.float32)

        # Ensure the samples have the same length
        min_len = min(master_sample.size, sample.size)
        master_sample_trimmed = master_sample[:min_len]
        sample_trimmed = sample[:min_len]

        # Calculate MSE
        mse = np.mean((master_sample_trimmed - sample_trimmed) ** 2)
        mse_values.append(mse)

        # Plot comparison sample
        times_sample = np.arange(sample_trimmed.size) / sr_sample
        plt.plot(times_sample, sample_trimmed, label=f'{audio_file.get_by_sample_rate_name()}', alpha=0.4)

    # Finalize plot
    plt.title('Comparison of Audio Samples')
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.show()

    # Print MSE values
    for path, mse in zip(sample_paths, mse_values):
        print(f'MSE for {path}: {mse}')


def plot_error_from_master(
        master_sample_path: AudioFile,
        sample_paths: list[AudioFile],
    ) -> None:
    # Load the master sample
    sr_master, master_sample = wavfile.read(master_sample_path.file_path)

    # Ensure master sample is in float format for precision
    master_sample = master_sample.astype(np.float32)

    mse_values = []

    # Set up the figure for plotting
    plt.figure(figsize=(14, len(sample_paths) * 2))

    for i, audio_file in enumerate(sample_paths, 1):
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
        plt.subplot(len(sample_paths), 1, i)
        times = np.arange(min_len) / sr_master
        plt.plot(times, error_signal, label=f'{audio_file.get_by_sample_rate_name()}')
        plt.title(f'Error from Master: {master_sample_path.file_path}')
        plt.xlabel('Time (s)')
        plt.ylabel('Error Amplitude')
        plt.legend()

    plt.tight_layout()
    plt.show()

    # Print MSE values
    for path, mse in zip(sample_paths, mse_values):
        print(f'MSE for {path}: {mse}')


def load_audio_for_comparison(master_path: AudioFile, sample_paths: list[AudioFile]):
    # Load and convert the master sample to a consistent format
    sr_master, master_sample = wavfile.read(master_path.file_path)
    master_sample = master_sample.astype(np.float32)

    errors = []
    mse_values = []

    print(sample_paths)
    for audio_file in sample_paths:
        path = audio_file.file_path
        # Determine the file type and load accordingly
        if path.endswith('.wav'):
            sr_sample, sample = wavfile.read(path)
            sample = sample.astype(np.float32)
        elif path.endswith('.mp3'):
            audio = AudioSegment.from_mp3(path)
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


def audio_similarity(file1, file2, sr=22050):
    """
    Compares two audio files (of potentially different codecs, sample rates, and bit depths)
    by computing the distance between their MFCCs.

    Parameters:
    - file1, file2: Paths to the audio files to compare.
    - sr: Target sample rate for both audio files.

    Returns:
    - Distance metric indicating similarity between the audio files.
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


def plot_mfcc_error(file1, file2, sr=22050):
    """
    Plots the error of MFCCs over time between two audio files.

    Parameters:
    - file1, file2: Paths to the audio files to compare.
    - sr: Target sample rate for both audio files.
    """
    # Load and normalize audio files
    y1, _ = librosa.load(file1, sr=sr)
    y2, _ = librosa.load(file2, sr=sr)

    y1 = librosa.util.normalize(y1)
    y2 = librosa.util.normalize(y2)

    # Compute MFCCs from the normalized audio signals
    mfcc1 = librosa.feature.mfcc(y=y1, sr=sr)
    mfcc2 = librosa.feature.mfcc(y=y2, sr=sr)

    # Assuming both MFCC matrices have the same shape; if not, additional handling is needed
    error = np.abs(mfcc1 - mfcc2)  # Absolute error between MFCCs

    # Plot the error
    plt.figure(figsize=(10, 4))
    plt.imshow(error, aspect='auto', origin='lower', interpolation='nearest')
    plt.title('MFCC Error Over Time')
    plt.xlabel('Time')
    plt.ylabel('MFCC Coefficients')
    plt.colorbar(label='Absolute Error')
    plt.tight_layout()
    plt.show()


def pad_mfcc(mfcc, target_shape):
    """
    Pads the MFCC array to the target shape with zeros.
    
    Parameters:
    - mfcc: The MFCC array to pad.
    - target_shape: The target shape tuple (n_mfcc, time_frames).
    
    Returns:
    - Padded MFCC array.
    """
    padding_width = target_shape[1] - mfcc.shape[1]
    if padding_width > 0:
        # Pad the array if it's shorter than the target shape
        return np.pad(mfcc, ((0, 0), (0, padding_width)), mode='constant', constant_values=0)
    else:
        # Return the array unmodified if it's already the target length or longer
        return mfcc


def compare_files_to_master(master_file: AudioFile, file_group: list[AudioFile], sr=22050):
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


if __name__ == "__main__":

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
        dataset = pd.DataFrame(columns=column_names)

        for sample_rate, files in by_sample_rate.items():
            sample_paths = sorted(files)

            # plot_error_from_master(AudioFile(file_path=master_sample_path), sample_paths)     ## NOT VERY USEFUL
            # compare_audio_samples(AudioFile(file_path=master_sample_path), sample_paths)

            # sample_paths.append(AudioFile(master_sample_path))
            compare_files_to_master(AudioFile(master_sample_path), sample_paths)
            plt.savefig(f'plots/{time_folder}/{folder}_S{sample_rate}.svg')
            plt.close()

            for file in sample_paths:
                distance = audio_similarity(master_sample_path, file.file_path)
                # print(distance)
                dataset.loc[file.sample_rate, file.get_by_sample_rate_name()] = distance
                # plot_mfcc_error(master_sample_path, file.file_path)

        ##### PLOT DATAFRAME #####
        dataset = dataset.sort_index()

        for column in dataset.columns:
            plt.plot(dataset.index, dataset[column], label=column)
        plt.title(f'{folder} MFCC Euclidean Distance By Sample Rate and Encoding')
        plt.xlabel('Sample Rate (Hz)')
        plt.ylabel('Euclidean Distance of the MFCC')
        plt.legend()
        plt.show()
        plt.savefig(f'plots/{time_folder}/{folder}_distance.svg')
        plt.close()
        ##########################
