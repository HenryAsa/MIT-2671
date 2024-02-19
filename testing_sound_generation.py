import numpy as np
import wavio
folder_name = "audio"


frequency = 440.0              # sound frequency (Hz)
sample_duration = 5                  # sample duration (seconds)
BITRATE_MAP = {1: 8, 2: 16, 3: 24}

def generate_single_frequency(frequency, sample_duration, sample_rate, sample_bit_rate):
    sample_number = int(sample_rate * sample_duration)          # number of samples
    time_vals = np.arange(sample_number)/sample_rate            # grid of time values
    x = np.sin(2*np.pi * frequency * time_vals)
    wavio.write(f'{folder_name}/sine_F{frequency}_S{sample_rate}_B{8 * sample_bit_rate}.wav', x, sample_rate, sampwidth=sample_bit_rate)

if __name__ == "__main__":
    frequency = 440
    sample_duration = 5

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