import os
from audio_file_rep import AudioFile
from audio_frequency_generation import generate_single_frequency_audio_files
from audio_modifications import modify_audio_sample
from constants import DATA_DIRECTORY
from datetime import datetime
import re
from scipy.io import wavfile

import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import numpy as np
from pydub import AudioSegment

from data_analysis import load_data_by_partitions
from utils import get_filetype_from_folder


def load_sample_audio_data(audio: AudioSegment):
    """
    Loads audio from the given file path and returns a numpy array of samples.
    Automatically handles different audio formats and bit depths.

    Args:
    filepath (str): The path to the audio file.

    Returns:
    numpy.ndarray: Numpy array containing the audio samples.
    """
    # Determine the sample width (in bytes) and set the appropriate dtype
    sample_width = audio.sample_width
    if sample_width == 1:
        # 8-bit audio - unsigned (since PyDub reads 8-bit audio as unsigned bytes)
        dtype = np.uint8
        max_val = 2**8 - 1
    elif sample_width == 2:
        # 16-bit audio - signed
        dtype = np.int16
        max_val = 2**15 - 1
    elif sample_width == 3:
        # 24-bit audio - needs special handling since numpy does not have 24-bit int types
        # Read as 32-bit int (np.int32) and handle manually if required
        dtype = np.int32
        max_val = 2**23 - 1
    elif sample_width == 4:
        # 32-bit audio - signed
        dtype = np.int32
        max_val = 2**31 - 1
    else:
        raise ValueError("Unsupported sample width")

    # Convert raw audio data to numpy array
    samples = np.frombuffer(audio.raw_data, dtype=dtype)

    # Normalize samples to be between -1 and 1
    samples = samples.astype(np.float32) / max_val

    # # Specific handling for 24-bit data:
    # if sample_width == 3:
    #     # Assume data is little-endian and stored as 24-bit packed in 32-bit integers
    #     samples = ((samples + 2**31) % 2**32 - 2**31) / (2**23 - 1)  # Adjust scale

    # If the audio has multiple channels, they will be interleaved
    if audio.channels > 1:
        samples = samples.reshape(-1, audio.channels)

    return samples


def normalize_data(data: np.ndarray):
    normalize_factor = 1
    if isinstance(data[0], np.int32):
        normalize_factor = 2**31
    elif isinstance(data[0], np.int16):
        data = data.astype(np.int32)
        normalize_factor = 2**15
    elif isinstance(data[0], np.uint8):
        normalize_factor = 2**7
        data = data.astype(np.int32)
        data = data - 2**7

    return data / normalize_factor


def plot_waveform(
        data: np.ndarray,
        sample_rate: int,
        frequency: float,
        num_waves: float,
        current_call_num: int,
        num_datasets: int,
        label: str = "",
        isMaster: bool = False
    ):
    time_length = data.shape[0] / sample_rate

    end_index = round(num_waves * sample_rate / frequency)

    time = np.linspace(0., time_length, data.shape[0])
    time_period = time[0:end_index]
    data_period = data[0:end_index]

    line_style = 'dotted' if isMaster else '-'

    colormap = get_cmap('viridis')
    color = colormap(current_call_num / num_datasets)

    plt.step(time_period, data_period, where='mid', label=f'Sample Rate = {sample_rate/1000} kHz' if label=="" else label, ls=line_style, color=color)
    # plt.step(time_period, data_period, where='mid', label=f'Sample Rate = {sample_rate/1000} kHz' if label=="" else label, ls=line_style)


if __name__ == "__main__":
    initial_time = datetime.now().strftime("%m-%d_%H-%M")
    output_directory = f'{DATA_DIRECTORY}/frequencies/{initial_time}'

    frequencies = [440]
    sample_duration = 5

    sample_rates = [1000, 4000, 8000, 20000, 44100, 48000, 88200, 96000, 192000]

    generate_single_frequency_audio_files(
        frequencies=frequencies,
        sample_rates=sample_rates,
        output_directory=output_directory,
    )

    for frequency in frequencies:
        sample_filepaths = set(get_filetype_from_folder(f'{output_directory}/F{frequency}', '.wav')).union(set(get_filetype_from_folder(f'{output_directory}/F{frequency}', '.mp3')))

        for sample_filepath in sample_filepaths:
            if not sample_filepath.endswith("B24.wav"):
                continue

            file_params = sample_filepath.split("/")[-1].split("_")
            sample_rate = int(file_params[2][1:])

            if sample_rate <= 48000:
                output_filename = re.sub(r'_B24.wav', '_BR.mp3', sample_filepath)
                modify_audio_sample(
                    audio_segment=AudioSegment.from_file(sample_filepath),
                    output_filename=output_filename,
                    output_file_format='.mp3',
                    output_sample_rate=sample_rate,
                    export_to_mp3=True,
                )

        sample_filepaths = set(get_filetype_from_folder(f'{output_directory}/F{frequency}', '.wav')).union(set(get_filetype_from_folder(f'{output_directory}/F{frequency}', '.mp3')))

        by_sample_rate: dict[int, set[AudioFile]] = {}
        by_file_type: dict[str, set[AudioFile]] = {}

        for file in sorted(sample_filepaths):
            audio_file: AudioSegment = AudioSegment.from_file(file)
            
            audio_data = load_sample_audio_data(audio_file)

            if file[-3:] not in by_file_type:
                by_file_type[file[-3:]] = set()
            by_file_type[file[-3:]].add(file)

            if audio_file.frame_rate not in by_sample_rate:
                by_sample_rate[audio_file.frame_rate] = set()
            by_sample_rate[audio_file.frame_rate].add(file)

            partitions = {
                "by_sample_rate":  by_sample_rate,
                "by_file_type":    by_file_type,
            }

        plot_files = [f'data/frequencies/{initial_time}/F440/sine_F{frequency}_S{sr}_B24.wav' for sr in sample_rates]
        for index, file in enumerate(plot_files):
            sample_rate, audio_file = wavfile.read(file)
            plot_waveform(
                normalize_data(audio_file),
                sample_rate=sample_rate,
                frequency=frequency,
                num_waves=1/16,
                current_call_num=index,
                num_datasets=len(plot_files)-1,
                # label=file,
            )

        plot_filepath = f'plots/frequencies/{initial_time}/F{frequency}'
        plt.title(f'Single-Frequency Waveform at f = {frequency} Hz')
        plt.legend()
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")
        os.makedirs(plot_filepath, exist_ok = True)
        plt.savefig(f'{plot_filepath}/sine_F{frequency}_by_SR.pdf')
        plt.show()
