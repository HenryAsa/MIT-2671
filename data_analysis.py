import librosa
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.io import wavfile
from scipy.spatial.distance import euclidean
from pydub import AudioSegment

from audio_file_rep import AudioFile
from constants import DATA_AUDIO_SAMPLES_DIRECTORY, DATA_DIRECTORY, DATA_RECORDED_SAMPLES_DIRECTORY
from utils import get_audio_params_from_filepath, get_filetype_from_folder


def load_data_by_paritions(
        time_folder_name: str,
    ) -> dict[str, dict]:

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



    # for folder in os.listdir(recorded_samples_folder_path):
    #     recorded_filepaths = set(get_filetype_from_folder(f'{recorded_samples_folder_path}/{folder}', '.wav')).union(set(get_filetype_from_folder(f'{recorded_samples_folder_path}/{folder}', '.mp3')))
    #     audio_samples_filepaths = set(get_filetype_from_folder(f'{audio_samples_folder_path}/{folder}', '.wav')).union(set(get_filetype_from_folder(f'{audio_samples_folder_path}/{folder}', '.mp3')))

    #     num_periods = 5
    #     plt.close()

    #     by_sample_rate: dict[int, set[AudioFile]] = {}
    #     by_file_type: dict[str, set[AudioFile]] = {}
    #     # masters: dict[str, AudioFile] = {}

    #     for filename in sorted(recorded_filepaths):
    #         print(filename)
    #         current_file = AudioFile(file_path=filename)

    #         if current_file.song_simple_name[:7] != RECORDED_SAMPLE_FILENAME_PREFIX:
    #             master = current_file

    #         if current_file.file_type not in by_file_type:
    #             by_file_type[current_file.file_type] = set()
    #         by_file_type[current_file.file_type].add(current_file)



if __name__ == "__main__":

    time_folder = '04-02_00-44'
    audio_recording_data = load_data_by_paritions(time_folder_name=time_folder)

    for folder, folder_partitions in audio_recording_data.items():
    
        master_sample_path: str = folder_partitions["master_filepath"]
        by_sample_rate: dict[int, list[AudioFile]] = folder_partitions["by_sample_rate"]

        for sample_rate, files in by_sample_rate.items():
            sample_paths = sorted(files)

    #         plot_error_from_master(AudioFile(file_path=master_sample_path), sample_paths)
    #         # compare_audio_samples(AudioFile(file_path=master_sample_path), sample_paths)

    #     plt.show()

