from typing import Type
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fftpack import fft
import numpy as np
import os
# from pylab import *

# filename = "audio/sine_F440_S192000_B24.wav"
# filename_cd = "audio/sine_F440_S44100_B24.wav"
# filename_bad = "audio/sine_F440_S1000_B8.wav"

def open_wav_file(filename: str):
    sample_rate, data = wavfile.read(filename)
    return sample_rate, data

def FFT_of_Wav_File(sample_rate, data, frequency):
    time_length = data.shape[0] / sample_rate
    plot_range = 0.005

    a = data.T # this is a two channel soundtrack, I get the first track
    c = fft(a) # create a list of complex number
    d = len(c) // 2  # you only need half of the fft list

    plt.title(f'FFT of Recordings with Different Sample and Bit Rates for f = {frequency} Hz')
    plt.plot(abs(c[:(d-1)]), label=f'Sample Rate = {sample_rate}')
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


def plot_waveform(sample_rate, data, frequency, num_waves):
    time_length = data.shape[0] / sample_rate

    end_index = round(num_waves * sample_rate / frequency)

    time = np.linspace(0., time_length, data.shape[0])
    time_period = time[0:end_index]
    data_period = data[0:end_index]

    plt.step(time_period, data_period, where='mid', label=f'Sample Rate = {sample_rate}')
    # plt.plot(time_period, data_period, 'o--', label=f'Sample Rate = {sample_rate}', color='grey', alpha=0.3)
    # plt.plot(time_period, data_period, label=f'Sample Rate = {sample_rate}')

    plt.title(f'Waveform of Recordings with Different Sample and Bit Rates for f = {frequency} Hz')
    plt.legend()
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    # plt.show()


def get_files_from_folder(folder: str) -> list[str]:
    return sorted([f'{folder}/{filename}' for filename in os.listdir(folder)])


if __name__ == "__main__":
    frequency = 440
    num_periods = 0.1
    plt.close()

    # FFT_of_Wav_File(sample_rate, data)
    data_folder = "generated_audio"
    for filename in get_files_from_folder(data_folder):
        print(filename)
        sample_rate, data = open_wav_file(f'{data_folder}/{filename}')
        data = normalize_data(data)
        plot_waveform(sample_rate, data, frequency, num_periods)
        # FFT_of_Wav_File(sample_rate, data, frequency)

    # sample_rate_cd, data_cd = open_wav_file(filename_cd)
    # plot_waveform(sample_rate_cd, data_cd, frequency, num_periods)

    # sample_rate_bad, data_bad = open_wav_file(filename_bad)
    # plot_waveform(sample_rate_bad, data_bad, frequency, num_periods)

    plt.show()