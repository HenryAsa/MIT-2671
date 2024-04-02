import numpy as np
import wavio
import os

from constants import TEST_BIT_DEPTHS, TEST_SAMPLE_RATES


def generate_single_frequency(
        frequency: float,
        sample_rate: int = 44100,
        bit_depth: int = 16,
        sample_duration: float = 5,
        filename_prefix: str = "sine",
        output_directory: str = "audio_test_samples",
    ):
    """Generates an audio file that contains a single sine wave
    (single frequency file)

    Parameters
    ----------
    frequency : float
        Frequency (in Hz) of the desired sine wave
    sample_rate : int, optional
        Sample rate (in samples/second) of the associated file, by
        default 44100
    bit_depth : int, optional
        Bit depth (in bits) for a single audio sample frame, by
        default 16
    sample_duration : float, optional
        Length of time (in seconds) of the desired sample, by default 5
    filename_prefix : str, optional
        Prefix prepended to each audio file's name, by default False
    output_directory : str, optional
        Name of the folder to output all of the generated audio files
        to, by default "audio_test_samples"
    """
    filename = f'{filename_prefix}_F{frequency}_S{sample_rate}_B{bit_depth}.wav'

    sample_number = int(sample_rate * sample_duration)          # number of samples
    time_vals = np.arange(sample_number)/sample_rate            # grid of time values
    x = np.sin(2*np.pi * frequency * time_vals)
    wavio.write(f'{output_directory}/{filename}', x, sample_rate, sampwidth=int(bit_depth/8))


def generate_single_frequency_audio_files(
        frequencies: list[float] = [440, 1000, 2000, 4000, 8000, 12000, 16000],
        sample_rates: list[int | float] = TEST_SAMPLE_RATES,
        bit_depths: list[int] = TEST_BIT_DEPTHS,
        sample_duration: float = 5,
        filename_prefix: str = "sine",
        output_directory: str = "audio_test_samples",
    ):
    """Generates a bunch of audio files with varied frequencies,
    sample rates, and bit depths

    Parameters
    ----------
    frequencies : list[float], optional
        Frequencies (in Hz) of the desired audio files, by default
        [440, 1000, 2000, 4000, 8000, 12000, 16000]
    sample_rates : list[int | float], optional
        Sample rates of the associated files, by default
        [1000, 2000, 4000, 8000, 16000, 44100, 82000, 96000, 192000]
    bit_depths : list[int], optional
        Bit depths (in bits) for the audio samples, by default
        [8, 16, 24]
    sample_duration : float, optional
        Length of time (in seconds) of each audio sample, by default 5
    filename_prefix : str, optional
        Prefix prepended to each audio file, by default "sine"
    output_directory : str, optional
        Name of the folder to output all of the generated audio files,
        by default "audio_test_samples"
    """
    for freq in frequencies:
        folder_name = f'F{freq}'
        for sample_rate in sample_rates:
            for bit_depth in bit_depths:
                if sample_rate <= 2*freq:
                    ## DON'T GENERATE FILES THAT WILL HAVE ALIASING
                    continue

                os.makedirs(f'{output_directory}/{folder_name}', exist_ok=True)
                generate_single_frequency(
                    frequency=freq,
                    sample_rate=sample_rate,
                    bit_depth=bit_depth,
                    sample_duration=sample_duration,
                    output_directory=f'{output_directory}/{folder_name}'
                )
                print(f'GENERATED    {filename_prefix}_F{freq}_S{sample_rate}_B{bit_depth}.wav')
