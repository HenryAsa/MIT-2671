from pathlib import Path
from typing import Type
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fftpack import fft
import numpy as np
import os
# from pylab import *
from utils import map_to_discrete


DATA_DIRECTORY = "data"
"""Default directory containing all of the raw data"""
RECORDED_BIT_DEPTH = 32
"""Default bit depth that the recordings were taken at"""


def open_wav_file(filename: str):
    sample_rate, data = wavfile.read(filename)
    return sample_rate, data


def FFT_of_Wav_File(data, sample_rate, frequency, bit_depth, isMaster: bool = False):
    time_length = data.shape[0] / sample_rate
    plot_range = 0.005

    a = data.T # this is a two channel soundtrack, I get the first track
    c = fft(a) # create a list of complex number
    d = len(c) // 2  # you only need half of the fft list

    line_style = 'dotted' if isMaster else '-'

    plt.title(f'FFT of Recordings with Different Sample and Bit Rates for f = {frequency} Hz')
    plt.plot(abs(c[:(d-1)]), label=f'Sample Rate = {sample_rate}', ls=line_style)
    plt.xlim([(1-plot_range) * frequency * time_length, (1+plot_range) * frequency * time_length])
    plt.legend()


def normalize_data(data: np.ndarray):
    normalize_factor = 1
    if isinstance(data[0], np.int32):
        normalize_factor = 2**31
    elif isinstance(data[0], np.int16):
        data = data.astype(np.int32)
        normalize_factor = 2**15
    elif isinstance(data[0], np.uint8):
        normalize_factor = 2**8
        data = data.astype(np.int32)
        data = data - 2**7

    return data / normalize_factor
    # print([min(data), max(data)], type(data[0]))
    # print(f'HERE\t{[min(map_to_discrete(data, [min(data), max(data)], normalize_factor, [-1, 1])), max(map_to_discrete(data, [min(data), max(data)], normalize_factor, [-1, 1]))]}')
    # return map_to_discrete(data, [min(data), max(data)], normalize_factor, [-1, 1])


def plot_waveform(data, sample_rate, frequency, bit_depth, num_waves, isMaster: bool = False):
    time_length = data.shape[0] / sample_rate

    end_index = round(num_waves * sample_rate / frequency)

    time = np.linspace(0., time_length, data.shape[0])
    time_period = time[0:end_index]
    data_period = data[0:end_index]

    line_style = 'dotted' if isMaster else '-'

    plt.step(time_period, data_period, where='mid', label=f'Sample Rate = {sample_rate}', ls=line_style)
    # plt.plot(time_period, data_period, 'o--', label=f'Sample Rate = {sample_rate}', color='grey', alpha=0.3)
    # plt.plot(time_period, data_period, label=f'Sample Rate = {sample_rate}')

    plt.title(f'Waveform of Recordings with Different Sample and Bit Rates for f = {frequency} Hz')
    plt.legend()
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    # plt.show()


def get_files_from_folder(folder: str) -> list[str]:
    return sorted([f'{folder}/{filename}' for filename in os.listdir(folder)])


def get_filetype_from_folder(folder: str, filetype: str) -> list[str]:
    return sorted(str(filepath) for filepath in Path(folder).rglob(f'*.{filetype[1:] if filetype.startswith(".") else filetype}'))


if __name__ == "__main__":

    time_folder = f'03-12_23-18'
    audio_samples = "audio_test_samples"
    recorded_samples = "recorded_samples"
    recorded_sample_file_prefix = "result_"

    recorded_samples_folder_path = f'{DATA_DIRECTORY}/{time_folder}/{recorded_samples}'
    audio_samples_folder_path = f'{DATA_DIRECTORY}/{time_folder}/{audio_samples}'

    recorded_filepaths = set(get_filetype_from_folder(recorded_samples_folder_path, '.wav'))
    audio_samples_filepaths = set(get_filetype_from_folder(audio_samples_folder_path, '.wav'))

    num_periods = 5
    plt.close()

    for filename in sorted(recorded_filepaths):
        file_params = filename.split(recorded_sample_file_prefix)[1].split("_")
        filetype = file_params[0]
        frequency = int(file_params[1][1:])
        sample_rate = int(file_params[2][1:])
        bit_depth = int(file_params[-1].split(".")[0][1:])
        print(filename)
        print(f'\tF = {frequency} Hz\tS = {sample_rate}\tB = {bit_depth}')


        sample_rate, data = open_wav_file(filename)
        data = normalize_data(data)
        plot_waveform(data, sample_rate, frequency, bit_depth, num_periods)
        
        original_audio_sample = filename.replace(recorded_samples, audio_samples).replace(recorded_sample_file_prefix, "")
        if original_audio_sample in audio_samples_filepaths:
            og_sample_rate, og_data = open_wav_file(original_audio_sample)
            og_data = normalize_data(og_data)
            plot_waveform(og_data, og_sample_rate, frequency, RECORDED_BIT_DEPTH, num_periods, isMaster=True)
        plt.show()

        FFT_of_Wav_File(data, sample_rate, frequency, bit_depth)
        FFT_of_Wav_File(og_data, og_sample_rate, frequency, RECORDED_BIT_DEPTH, isMaster=True)
        plt.show()

    # sample_rate_cd, data_cd = open_wav_file(filename_cd)
    # plot_waveform(sample_rate_cd, data_cd, frequency, num_periods)

    # sample_rate_bad, data_bad = open_wav_file(filename_bad)
    # plot_waveform(sample_rate_bad, data_bad, frequency, num_periods)

    plt.show()