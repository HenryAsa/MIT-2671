
import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft
from scipy.io import wavfile
from pydub import AudioSegment
import os


## HELPER FUNCTIONS
def get_files_from_folder(folder: str) -> list[str]:
    return sorted([f'{folder}/{filename}' for filename in os.listdir(folder)])


def fft_plot(audio, sample_rate):
    ## https://stackoverflow.com/questions/74063787/how-to-plot-frequency-data-from-a-wav-file-in-python
    N = len(audio)    # Number of samples
    T = 1/sample_rate # Period
    y_freq = fft(audio)
    domain = len(y_freq) // 2
    x_freq = np.linspace(0, sample_rate//2, N//2)
    plt.plot(x_freq, abs(y_freq[:domain]))
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Frequency Amplitude |X(t)|")
    return plt.show()


if __name__ == "__main__":

    dsd_folder = 'dsd_audio'

    for dsd_filename in get_files_from_folder(dsd_folder):
        dsd_file: AudioSegment = AudioSegment.from_file(dsd_filename, "dsf")
        # print(dsd_file.raw_data)
        dsd_data = dsd_file.get_array_of_samples()

        print(dsd_data[::5000].tolist())
        print("Here")
        ## COMPUTER KILLER ##
        # fft_plot(dsd_data[::5000].tolist(), dsd_file.frame_rate)
        #####################

        # print(dsd_file)
