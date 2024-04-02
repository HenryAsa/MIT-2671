import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os

from audio_file_rep import AudioFile
from constants import DATA_AUDIO_SAMPLES_DIRECTORY, DATA_DIRECTORY, DATA_RECORDED_SAMPLES_DIRECTORY
from utils import get_audio_params_from_filepath, get_filetype_from_folder


def compare_audio_samples(
        master_sample_path: str,
        sample_paths: list[str]
    ) -> None:
    pass


if __name__ == "__main__":

    time_folder = '04-02_00-44'

    recorded_samples_folder_path = f'{DATA_DIRECTORY}/{time_folder}/{DATA_RECORDED_SAMPLES_DIRECTORY}'
    audio_samples_folder_path = f'{DATA_DIRECTORY}/{time_folder}/{DATA_AUDIO_SAMPLES_DIRECTORY}'

    for folder in os.listdir(recorded_samples_folder_path):
        recorded_filepaths = set(get_filetype_from_folder(f'{recorded_samples_folder_path}/{folder}', '.wav')).union(set(get_filetype_from_folder(f'{recorded_samples_folder_path}/{folder}', '.mp3')))
        audio_samples_filepaths = set(get_filetype_from_folder(f'{audio_samples_folder_path}/{folder}', '.wav')).union(set(get_filetype_from_folder(f'{audio_samples_folder_path}/{folder}', '.mp3')))

        by_sample_rate: dict[int, set[AudioFile]] = {}
        by_file_type: dict[str, set[AudioFile]] = {}
        # masters: dict[str, AudioFile] = {}

        master_sample_path = next((audio_file for audio_file in recorded_filepaths if audio_file.endswith("_S192000_B24.wav")))
        recorded_filepaths.discard(master_sample_path)

        for filename in sorted(recorded_filepaths):
            print(filename)
            current_file = AudioFile(file_path=filename)

            if current_file.file_type not in by_file_type:
                by_file_type[current_file.file_type] = set()
            by_file_type[current_file.file_type].add(current_file)

            if current_file.sample_rate not in by_sample_rate:
                by_sample_rate[current_file.sample_rate] = set()
            by_sample_rate[current_file.sample_rate].add(current_file)


        for sample_rate, files in by_sample_rate.items():
            sample_paths = sorted(files)
            compare_audio_samples(master_sample_path, sample_paths)


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

    #         if current_file.sample_rate not in by_sample_rate:
    #             by_sample_rate[current_file.sample_rate] = set()
    #         by_sample_rate[current_file.sample_rate].add(current_file)

    #     for sample_rate, files in by_sample_rate.items():
    #         plt.figure(figsize=[15,5])
    #         # ## LABELS FOR STANDARD FFT
    #         # plt.xlabel('Frequency (Hz)')
    #         # plt.ylabel('Amplitude')
    #         # plt.title(f'FFT of the Audio Files for Sampling Rate of {sample_rate} Hz')
    #         ## LABELS FOR FFT DIFF
    #         plt.xlabel('Frequency (Hz)')
    #         plt.ylabel('Discrepancy in Amplitude')
    #         plt.title(f'FFT Discrepancies between Master and Sample for Sampling Rate of {sample_rate} Hz')

    #         master_fft = get_fft(master.file_path)

    #         for audio_file in files:
    #             filename = audio_file.file_path
                
    #             if filename == master.file_path:
    #                 continue

    #             # plot_FFT(audio_file=audio_file, value_name=audio_file.get_by_sample_rate_name(), isMaster=False)
    #             plot_FFT_diff(audio_file=audio_file, value_name=audio_file.get_by_sample_rate_name(), master_fft=master_fft)

    #         ## PLOT MASTER
    #         # plot_FFT(audio_file=master, value_name="Master", isMaster=True)
    #         plot_FFT_diff(audio_file=master, value_name="Master", master_fft=master_fft)

    #         plt.grid()
    #         plt.xlim(0, 5000)
    #         plt.show()

    #     plt.show()

