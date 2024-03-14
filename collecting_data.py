from datetime import datetime
import time
import matplotlib.pyplot as plt
import os
import sounddevice as sd
from pathlib import Path
import queue
import soundfile as sf
import sys

from signals import Signal, Sine
from audio_generation import generate_audio_files

DATA_DIRECTORY = "data"
"""Default directory to store all of the raw data files"""
RECORDING_SAMPLE_RATE = 96000
"""Default sampling rate of 96 kHz for recorded samples"""


q = queue.Queue()

def callback_in(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    if status:
        print(status, file=sys.stderr)
    q.put(indata.copy())


def read_simple_audio_samples(
        audio_samples_directory: str = "audio_test_samples",
        output_directory: str = "generated_audio"
    ):

    print(sd.query_devices())

    print(f'INPUT SETTINGS: \t{sd.check_input_settings()}')
    print(f'OUTPUT SETTINGS:\t{sd.check_output_settings()}')

    audio_files = sorted(str(filename) for filename in Path(audio_samples_directory).rglob('*.wav'))
    os.makedirs(output_directory, exist_ok=True)

    for filename in audio_files:
        # READ AUDIO FILES
        file_params = str(filename).split("_")
        filetype = file_params[2].split("/")[-1]
        freq = int(file_params[3][1:])
        sample_rate = int(file_params[4][1:])
        bit_depth = int(file_params[-1].split(".")[0][1:])
        output_filename = f'result_{filetype}_F{freq}_S{sample_rate}_B{bit_depth}.wav'

        os.makedirs(f'{output_directory}/F{freq}', exist_ok=True)

        simultaneous_record_playback(
            input_filename=filename,
            sample_rate=sample_rate,
            bit_depth=bit_depth,
            output_directory=f'{output_directory}/F{freq}',
            output_filename=output_filename,
        )


def simultaneous_record_playback(
        input_filename: str,
        sample_rate: int,
        bit_depth: int,
        output_directory: str,
        output_filename: str = False,
    ):

    sd.default.device = 3, 2

    print(input_filename)

    if output_filename is False:
        output_filename = f'result_{input_filename.split(".")[0]}_S{sample_rate}_B{bit_depth}.wav'

    audio_data, sample_rate = sf.read(input_filename)
    duration = len(audio_data)/sample_rate
    print(f'\tS = {sample_rate}\tB = {bit_depth}\tDURATION = {duration}')

    with sf.SoundFile(
            f'{output_directory}/{output_filename}',
            mode='w+',
            samplerate=RECORDING_SAMPLE_RATE,
            channels=audio_data.ndim,
            subtype='PCM_32',
            format='WAV'
        ) as file:
        with sd.InputStream(
                samplerate=RECORDING_SAMPLE_RATE,
                channels=1,
                callback=callback_in
            ):
            sd.play(data = audio_data, samplerate=sample_rate, blocking=False)
            start_time = time.time()
            while time.time() - start_time <= duration:
                file.write(q.get())

    sd.stop()


if __name__ == "__main__":
    initial_time = datetime.now().strftime("%m-%d_%H-%M")
    # shared_output_directory = f'{DATA_DIRECTORY}/{initial_time}'

    # generate_audio_files(output_directory=f'{shared_output_directory}/audio_test_samples')
    # read_simple_audio_samples(audio_samples_directory="audio_test_samples", output_directory=f'{shared_output_directory}/recorded_samples')
    read_simple_audio_samples(audio_samples_directory="music", output_directory=f'music')
