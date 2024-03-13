import time
import numpy as np
import wavio
import matplotlib.pyplot as plt
import os
import sounddevice as sd
from pathlib import Path
import queue
import soundfile as sf
import sys



from signals import Signal, Sine

folder_name = "generated_audio"


frequency = 440.0                       # sound frequency (Hz)
sample_duration = 5                     # sample duration (seconds)

def generate_single_frequency(frequency, sample_duration, sample_rate, sample_bit_rate, folder_name):
    sample_number = int(sample_rate * sample_duration)          # number of samples
    time_vals = np.arange(sample_number)/sample_rate            # grid of time values
    x = np.sin(2*np.pi * frequency * time_vals)
    wavio.write(f'{folder_name}/sine_F{frequency}_S{sample_rate}_B{sample_bit_rate}.wav', x, sample_rate, sampwidth=int(sample_bit_rate/8))



def generate_audio_files():
    frequencies = [440, 1000, 2000, 4000, 8000, 12000, 16000]
    sample_rates = [1000, 2000, 4000, 8000, 16000, 44100, 82000, 96000, 192000]
    bitrates = [8, 16, 24]
    sample_duration = 5

    for freq in frequencies:
        folder_name = f'F{freq}'
        for sample_rate in sample_rates:
            for bitrate in bitrates:
                if sample_rate <= 2*freq:
                    ## ALIASING IS OCCURING
                    continue
                os.makedirs(f'audio_test_samples/{folder_name}', exist_ok=True)
                generate_single_frequency(freq, sample_duration, sample_rate, bitrate, f'audio_test_samples/{folder_name}')
                print(f'sine_F{frequency}_S{sample_rate}_B{bitrate}.wav')


q = queue.Queue()

def callback_in(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    if status:
        print(status, file=sys.stderr)
    q.put(indata.copy())


def record_audio_samples():

    RECORDING_SAMPLE_RATE = 96000

    print(f'INPUT SETTINGS: \t{sd.check_input_settings()}')
    print(f'OUTPUT SETTINGS:\t{sd.check_output_settings()}')

    audio_files = list(Path("audio_test_samples").rglob("*.wav"))
    output_folder = f'recorded_samples'
    os.makedirs(output_folder, exist_ok=True)

    for filename in audio_files:
        # read MP3 file
        file_params = str(filename).split("_")
        filetype = file_params[2].split("/")[-1]
        freq = file_params[3][1:]
        sample_rate = file_params[4][1:]
        bit_depth = file_params[-1].split(".")[0][1:]
        
        output_filename = f'result_{filetype}_F{freq}_S{sample_rate}_B{bit_depth}.wav'
        os.makedirs(f'{output_folder}/F{freq}', exist_ok=True)
        print(filename)
        

        filename = str(filename)
        sound_data, _fs = sf.read(filename)
        duration = len(sound_data)/_fs
        print(f'\tF = {freq} Hz\tS = {sample_rate}\tB = {bit_depth}\tDURATION = {duration}')


        with sf.SoundFile(f'{output_folder}/F{freq}/{output_filename}', mode='w+', samplerate=RECORDING_SAMPLE_RATE,
                      channels=1, subtype='PCM_32', format='WAV') as file:
            with sd.InputStream(samplerate=RECORDING_SAMPLE_RATE, channels=1, callback=callback_in):
                sd.play(data = sound_data, samplerate=_fs, blocking=False)
                start_time = time.time()
                while time.time() - start_time <= duration:
                    file.write(q.get())

        sd.stop()


if __name__ == "__main__":
    record_audio_samples()
    generate_audio_files()
    # frequency = 440
    # sample_duration = 5

    # test_signal_1000_8 = Sine(frequency=frequency, time_length=sample_duration, sample_rate=1000, bitrate=8)
    # test_signal_44100_8 = Sine(frequency=frequency, time_length=sample_duration, sample_rate=44100, bitrate=8)
    # test_signal_44100_16 = Sine(frequency=frequency, time_length=sample_duration, sample_rate=44100, bitrate=16)

    # test_signal_1000_8_audio = Sine.from_wav(f'{folder_name}/sine_F{frequency}_S1000_B8.wav', bitrate=8, frequency=frequency)
    # test_signal_1000_8_discrete = Sine(frequency=frequency, time_length=sample_duration, sample_rate=1000, bitrate=8, discretize = True)


    # num_waves_to_plot = 5

    # test_signal_1000_8.plot_waveform(num_waves=num_waves_to_plot)
    # test_signal_1000_8_audio.plot_waveform(num_waves=num_waves_to_plot)
    # test_signal_1000_8_discrete.plot_waveform(num_waves=num_waves_to_plot)
    # # test_signal_44100_8.plot_waveform(num_waves=num_waves_to_plot)
    # # test_signal_44100_16.plot_waveform(num_waves=num_waves_to_plot)

    # plt.show()
    raise TypeError
