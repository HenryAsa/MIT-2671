from datetime import datetime
import time
import matplotlib.pyplot as plt
import os
import sounddevice as sd
from pathlib import Path
import queue
import soundfile as sf
import sys

from constants import DATA_DIRECTORY, RECORDING_SAMPLE_RATE
from signals import Signal, Sine
from audio_generation import generate_audio_files



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
    """
    Process and categorize audio samples from a specified directory.

    This function reads audio files from a given directory, extracts
    their parameters based on their filenames, and uses these
    parameters to play and record the audio samples simultaneously
    using the 'simultaneous_record_playback' function. The resulting
    audio files are saved into a structured directory format based on
    their frequency.

    Parameters
    ----------
    audio_samples_directory : str, optional
        The directory containing the audio sample files to be
        processed. These files should be named in a specific format to
        extract parameters like frequency, sample rate, and bit depth.
        The default directory is "audio_test_samples".
    output_directory : str, optional
        The base directory where the processed audio files will be
        saved. The files are organized into subdirectories based on
        their frequencies. The default directory is "generated_audio".

    Notes
    -----
    The audio sample filenames in the source directory should follow a
    specific naming convention to allow correct extraction of
    parameters: for example, "audio_F440_S44100_B16.wav", where F
    represents the frequency, S the sample rate, and B the bit depth.

    This function utilizes the 'simultaneous_record_playback' function
    to play and record the audio samples. It also creates necessary
    directories based on the audio frequency extracted from the file
    names.

    It is recommended to check the input and output settings of the
    sound device before running this function, as incorrect settings
    might affect the recording and playback quality.

    Examples
    --------
    >>> read_simple_audio_samples(
            audio_samples_directory='my_audio_samples',
            output_directory='my_processed_audio'
        )
    This will read all WAV files from 'my_audio_samples', process
    them, and save the results under 'my_processed_audio', categorized
    by frequency.
    """
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
    """
    Record audio input and play back an audio file simultaneously.

    This function plays back an audio file while simultaneously
    recording audio input.  The recorded audio is then saved to a
    specified directory. The function allows setting of sample rate
    and bit depth for the recording. The output filename is optional;
    if not provided, it is generated automatically based on input
    filename, sample rate, and bit depth.

    Parameters
    ----------
    input_filename : str
        The path to the audio file to be played back during the
        recording.
    sample_rate : int
        The sample rate in Hz for the audio recording and playback.
    bit_depth : int
        The bit depth for the recorded audio. Typically 16, 24, or 32.
    output_directory : str
        The directory where the recorded audio file will be saved.
    output_filename : str, optional
        The name of the output file to save the recorded audio. If not
        specified, a filename is generated automatically based on the
        input file name, sample rate, and bit depth. Defaults to
        False, which triggers auto-naming.

    Notes
    -----
    The function sets the sound device's default input and output
    channels before starting the playback and recording processes. The
    recording duration is equal to the duration of the input audio
    file.

    Callback function for the input stream and queue for handling
    audio frames must be defined outside this function.

    The recorded audio format is set to PCM 32-bit WAV.

    Examples
    --------
    >>> simultaneous_record_playback(
            input_filename='input.wav',
            sample_rate=44100,
            bit_depth=24,
            output_directory='./recordings'
        )
    This will play 'input.wav' and record simultaneously, saving the recorded audio in 
    './recordings' directory with an automatically generated filename.
    """

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
