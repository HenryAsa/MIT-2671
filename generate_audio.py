import numpy as np
import wavio
import os

def generate_single_frequency(
        frequency: float,
        sample_duration: float = 5,
        sample_rate: int = 44100,
        bit_depth: int = 16,
        folder_name: str = "audio_test_samples",
        filename: str = False
    ):
    if filename is False:
        filename = f'sine_F{frequency}_S{sample_rate}_B{bit_depth}.wav'

    sample_number = int(sample_rate * sample_duration)          # number of samples
    time_vals = np.arange(sample_number)/sample_rate            # grid of time values
    x = np.sin(2*np.pi * frequency * time_vals)
    wavio.write(f'{folder_name}/{filename}', x, sample_rate, sampwidth=int(bit_depth/8))


def generate_audio_files(
        frequencies: list[float] = [440, 1000, 2000, 4000, 8000, 12000, 16000],
        sample_rates: list[int | float] = [1000, 2000, 4000, 8000, 16000, 44100, 82000, 96000, 192000],
        bit_depths: list[int] = [8, 16, 24],
        sample_duration: float = 5,
        filename_prefix: str = "sine",
        output_directory: str = "audio_test_samples",
    ):
    for freq in frequencies:
        folder_name = f'F{freq}'
        for sample_rate in sample_rates:
            for bit_depth in bit_depths:
                if sample_rate <= 2*freq:
                    ## DON'T GENERATE FILES THAT WILL HAVE ALIASING
                    continue

                os.makedirs(f'{output_directory}/{folder_name}', exist_ok=True)
                generate_single_frequency(freq, sample_duration, sample_rate, bit_depth, f'{output_directory}/{folder_name}')
                print(f'GENERATED    {filename_prefix}_F{freq}_S{sample_rate}_B{bit_depth}.wav')
