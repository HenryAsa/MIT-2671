import numpy as np
import wavio
import matplotlib.pyplot as plt

from signals import Signal, Sine

folder_name = "generated_audio"


frequency = 440.0                       # sound frequency (Hz)
sample_duration = 5                     # sample duration (seconds)

def generate_single_frequency(frequency, sample_duration, sample_rate, sample_bit_rate):
    sample_number = int(sample_rate * sample_duration)          # number of samples
    time_vals = np.arange(sample_number)/sample_rate            # grid of time values
    x = np.sin(2*np.pi * frequency * time_vals)
    wavio.write(f'{folder_name}/sine_F{frequency}_S{sample_rate}_B{8 * sample_bit_rate}.wav', x, sample_rate, sampwidth=sample_bit_rate)

if __name__ == "__main__":
    frequency = 440
    sample_duration = 5

    test_signal_1000 = Sine(frequency=frequency, time_length=sample_duration, sample_rate=1000, bitrate=8)
    test_signal_44100 = Sine(frequency=frequency, time_length=sample_duration, sample_rate=44100, bitrate=8)

    num_waves_to_plot = 5

    test_signal_1000.plot_waveform(num_waves=num_waves_to_plot)
    test_signal_44100.plot_waveform(num_waves=num_waves_to_plot)

    plt.show()
    raise TypeError

    sample_rate = 1000
    sample_bit_rate = 1
    generate_single_frequency(frequency, sample_duration, sample_rate, sample_bit_rate)

    sample_rate = 44100
    sample_bit_rate = 2
    generate_single_frequency(frequency, sample_duration, sample_rate, sample_bit_rate)

    sample_rate = 88200
    sample_bit_rate = 3
    generate_single_frequency(frequency, sample_duration, sample_rate, sample_bit_rate)

    sample_rate = 96000
    sample_bit_rate = 3
    generate_single_frequency(frequency, sample_duration, sample_rate, sample_bit_rate)

    sample_rate = 192000
    sample_bit_rate = 3
    generate_single_frequency(frequency, sample_duration, sample_rate, sample_bit_rate)