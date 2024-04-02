from random import sample
from typing import Type
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fftpack import fft
import numpy as np
import os

from utils import get_audio_params_from_filepath, get_filetype_from_folder, map_to_discrete
from constants import DATA_AUDIO_SAMPLES_DIRECTORY, DATA_DIRECTORY, DATA_RECORDED_SAMPLES_DIRECTORY, RECORDED_BIT_DEPTH, RECORDED_SAMPLE_FILENAME_PREFIX


class AudioFile:
    def __init__(self, file_path) -> None:
        self.file_path = file_path
        file_params = get_audio_params_from_filepath(filepath=file_path)
        self.file_type = file_params["filetype"]
        self.song_simple_name = file_params["song_simple_name"]
        self.output_file_name = file_params["output_filename"]
        self.sample_rate = file_params["sample_rate"]

        self.bit_depth = file_params["bit_depth"] if self.file_type in {'.flac', '.wav'} else None
        self.bitrate = file_params["bitrate"] if self.file_type == '.mp3' else None

    # @property
    # def bitrate(self):
    #     assert self.file_type == '.mp3', f'bitrate is only attainable for .mp3 files, but this file is of type {self.file_type}'
    #     return self.bitrate

    # @property
    # def bit_depth(self):
    #     assert self.file_type in {'.flac', '.wav'}, f'bit depth is only attainable for .flac and .wav files, but this file is of type {self.file_type}'
        # return self.bit_depth

    def get_by_sample_rate_name(self):
        if self.file_type == '.mp3':
            return f'MP3: BitRate = {self.bitrate} kb/s'
        else:
            return f'WAV: Bit Depth = {self.bit_depth}'

    def get_by_file_type_name(self):
        if self.file_type == '.mp3':
            return f'Sample Rate = {self.sample_rate} Hz, BitRate = {self.bitrate} kb/s'
        else:
            return f'Sample Rate = {self.sample_rate} Hz, Bit Depth = {self.bit_depth}'


def open_wav_file(filename: str):
    sample_rate, data = wavfile.read(filename)
    return sample_rate, data


def get_fft(file_path: str):
    sample_rate, data = wavfile.read(file_path)

    if len(data.shape) > 1:
        print("USING THE FIRST CHANNEL FOR THIS FILE")
        data = data[:, 0]  # Take the first channel for the master file

    fft_data = np.fft.fft(data)
    fft_freq = np.fft.fftfreq(len(fft_data), 1/sample_rate)

    return fft_data, fft_freq


def plot_FFT_diff(audio_file: AudioFile, value_name: str, master_fft: tuple):
    master_fft_data, master_fft_freq = master_fft

    sample_fft_data, sample_fft_freq = get_fft(audio_file.file_path)

    # Determine the minimum length of the two arrays
    min_length = min(len(master_fft_data), len(sample_fft_data)) // 2

    # Truncate both arrays to the minimum length
    truncated_master_fft_data = np.abs(master_fft_data)[:min_length]
    truncated_sample_fft_data = np.abs(sample_fft_data)[:min_length]

    fft_diff = truncated_master_fft_data - truncated_sample_fft_data
    # fft_diff = np.abs(master_fft_data)[:len(master_fft_data)//2] - np.abs(sample_fft_data)[:len(sample_fft_data)//2]

    # Plot the discrepancies
    # plt.plot(sample_fft_data, label=value_name)
    plt.plot(master_fft_freq[:len(master_fft_data)//2], fft_diff, label=value_name)  # Plotting only the positive half of the frequencies
    plt.legend()
    # plt.grid()
    # plt.show()



def plot_FFT(audio_file: AudioFile, value_name: str, isMaster: bool = False):

    # Compute the FFT
    fft_data, fft_freq = get_fft(audio_file.file_path)

    # Plot the FFT
    # plt.plot(fft_data, label=value_name)
    plt.plot(fft_freq[:len(fft_data)//2], np.abs(fft_data)[:len(fft_data)//2], label=value_name)  # Plotting only the positive half of the frequencies
    plt.legend()

    # plt.title(f'FFT of Recordings with Different Sample and Bit Rates for f = {frequency} Hz')
    # plt.plot(abs(c[:(d-1)]), label=f'Sample Rate = {sample_rate}', ls=line_style)
    # plt.xlim([(1-plot_range) * frequency * time_length, (1+plot_range) * frequency * time_length])


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


def plot_waveform(
        data,
        sample_rate,
        frequency,
        bit_depth,
        num_waves,
        isMaster: bool = False
    ):
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


if __name__ == "__main__":

    time_folder = f'03-14_22-47'

    recorded_samples_folder_path = f'{DATA_DIRECTORY}/{time_folder}/{DATA_RECORDED_SAMPLES_DIRECTORY}'
    audio_samples_folder_path = f'{DATA_DIRECTORY}/{time_folder}/{DATA_AUDIO_SAMPLES_DIRECTORY}'

    for folder in os.listdir(recorded_samples_folder_path):
        recorded_filepaths = set(get_filetype_from_folder(f'{recorded_samples_folder_path}/{folder}', '.wav')).union(set(get_filetype_from_folder(f'{recorded_samples_folder_path}/{folder}', '.mp3')))
        audio_samples_filepaths = set(get_filetype_from_folder(f'{audio_samples_folder_path}/{folder}', '.wav')).union(set(get_filetype_from_folder(f'{audio_samples_folder_path}/{folder}', '.mp3')))

        num_periods = 5
        plt.close()

        by_sample_rate: dict[int, set[AudioFile]] = {}
        by_file_type: dict[str, set[AudioFile]] = {}
        # masters: dict[str, AudioFile] = {}

        for filename in sorted(recorded_filepaths):
            print(filename)
            current_file = AudioFile(file_path=filename)

            if current_file.song_simple_name[:7] != RECORDED_SAMPLE_FILENAME_PREFIX:
                master = current_file

            if current_file.file_type not in by_file_type:
                by_file_type[current_file.file_type] = set()
            by_file_type[current_file.file_type].add(current_file)

            if current_file.sample_rate not in by_sample_rate:
                by_sample_rate[current_file.sample_rate] = set()
            by_sample_rate[current_file.sample_rate].add(current_file)

        for sample_rate, files in by_sample_rate.items():
            plt.figure(figsize=[15,5])
            # ## LABELS FOR STANDARD FFT
            # plt.xlabel('Frequency (Hz)')
            # plt.ylabel('Amplitude')
            # plt.title(f'FFT of the Audio Files for Sampling Rate of {sample_rate} Hz')
            ## LABELS FOR FFT DIFF
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Discrepancy in Amplitude')
            plt.title(f'FFT Discrepancies between Master and Sample for Sampling Rate of {sample_rate} Hz')

            master_fft = get_fft(master.file_path)

            for audio_file in files:
                filename = audio_file.file_path
                
                if filename == master.file_path:
                    continue

                # plot_FFT(audio_file=audio_file, value_name=audio_file.get_by_sample_rate_name(), isMaster=False)
                plot_FFT_diff(audio_file=audio_file, value_name=audio_file.get_by_sample_rate_name(), master_fft=master_fft)

            ## PLOT MASTER
            # plot_FFT(audio_file=master, value_name="Master", isMaster=True)
            plot_FFT_diff(audio_file=master, value_name="Master", master_fft=master_fft)

            plt.grid()
            plt.xlim(0, 5000)
            plt.show()

        plt.show()