from datetime import datetime
import os
from pydub import AudioSegment
from constants import DATA_AUDIO_SAMPLES_DIRECTORY, DATA_DIRECTORY, MP3_BITRATES, RECORDED_SAMPLE_FILENAME_PREFIX
from collecting_data import simultaneous_record_playback
import numpy as np
import soundfile as sf



def crop_audio(input_file: str, start_ms: int, end_ms: int, output_directory: str, output_filename: str) -> AudioSegment:
    """
    Crop the given audio file between the specified start and end
    times and saves it

    Parameters
    ----------
    input_file : str
        The path to the input .flac or .mp3 or .wav audio file
    start_ms : int
        The start time in milliseconds from which to start cropping
    end_ms : int
        The end time in milliseconds at which to stop cropping
    output_directory : str
        Directory to save the cropped audio file in
    output_filename : str
        The path to the output .flac or .mp3 or .wav audio file
    output_sample_rate : int
        The sample rate at which to encode the final output file, by
        default False indicating that the sample rate should not be
        changed


    Returns
    -------
    pydub.AudioSegment
        Cropped audio segment in the format that it has been modified
        to

    Raises
    ------
    ValueError
        If the input file format is not supported (only .flac and .wav
        files are supported)

    Examples
    --------
    >>> crop_audio('example.flac', 1000, 5000, 'cropped_example.flac')
    This will crop the 'example.flac' from 1 second to 5 seconds and
    save the cropped audio to 'cropped_example.flac'.
    """
    # Determine the format based on the file extension
    file_format = input_file.split('.')[-1]
    if file_format not in {'flac', 'mp3', 'wav'}:
        raise ValueError("Unsupported file format: Only .flac and .wav files are supported.")

    # Load the audio file
    audio: AudioSegment = AudioSegment.from_file(input_file, format=file_format)

    cropped_audio: AudioSegment = audio[start_ms:end_ms]
    os.makedirs(output_directory, exist_ok=True)
    cropped_audio.export(f'{output_directory}/{output_filename}', format=file_format if file_format != 'flac' else 'wav')

    return cropped_audio


def modify_audio_sample(
        audio_segment: AudioSegment,
        output_filename: str,
        output_file_format: str,
        output_sample_rate: int = False,
        export_to_mp3: bool = False,
        output_bit_depth: int = False,
    ):
    if output_filename.find(".") != -1:
        assert output_filename.split(".")[-1] == output_file_format.split(".")[-1], f'expected output_filename to have same format as output_file_format, but "{output_filename.split(".")[-1]}" is not the same as "{output_file_format.split(".")[-1]}"'

    if output_sample_rate is not False:
        audio_segment.set_frame_rate(output_sample_rate)

    # Save the cropped audio file
    if output_bit_depth is not False:
        audio_segment.export(out_f=output_filename, format=output_file_format.replace(".", ""), bitrate=str(int(output_bit_depth/8)))

    if export_to_mp3 is not False:
        for mp3_bitrate in MP3_BITRATES:
            audio_segment.export(out_f=output_filename.replace('_BR.mp3', f'_BR{mp3_bitrate}.mp3'), format='mp3', bitrate=mp3_bitrate)


def remaster_audio_file(
        audio_name: str,
        original_audio: AudioSegment,
        output_directory: str,
        make_mp3_files: bool = True,
    ) -> None:
    sample_output_directory = f'{output_directory}/{audio_name}'
    os.makedirs(sample_output_directory, exist_ok=True)

    for sample_rate in [1000, 2000, 4000, 8000, 16000, 44100, 48000, 82000, 96000, 192000]:
        for bit_depth in [8, 16, 24]:
            output_filename = f'{RECORDED_SAMPLE_FILENAME_PREFIX}{audio_name}_S{sample_rate}_B{bit_depth}.wav'
            modify_audio_sample(
                audio_segment=original_audio,
                output_filename=f'{sample_output_directory}/{output_filename}',
                output_file_format='.wav',
                output_sample_rate=sample_rate,
                output_bit_depth=bit_depth,
            )

    if make_mp3_files is True:
        output_filename = f'{RECORDED_SAMPLE_FILENAME_PREFIX}{audio_name}_BR.mp3'
        modify_audio_sample(
            audio_segment=original_audio,
            output_filename=f'{sample_output_directory}/{output_filename}',
            output_file_format='.mp3',
            output_sample_rate=sample_rate,
            output_bit_depth=bit_depth,
        )


if __name__ == "__main__":

    initial_time = datetime.now().strftime("%m-%d_%H-%M")
    samples_output_directory = f'{DATA_DIRECTORY}/{initial_time}/{DATA_AUDIO_SAMPLES_DIRECTORY}'

    audio_samples = {
        "hotel_california_intro": {
            "start_ms": 0,
            "end_ms": 9000,
            "filepath": "music/1-Hotel California.flac",
        },
        "hotel_california_guitar_solo": {
            "start_ms": 328000,
            "end_ms": 333000,
            "filepath": "music/1-Hotel California.flac",
        }
    }

    for audio_name, audio_specs in audio_samples.items():
        cropped_audio_sample = crop_audio(
            input_file=audio_specs["filepath"],
            start_ms=audio_specs["start_ms"],
            end_ms=audio_specs["end_ms"],
            output_directory=f'{samples_output_directory}/{audio_name}',
            output_filename=f'{audio_name}_S192000_B24.wav',
        )

        remaster_audio_file(
            audio_name=audio_name,
            original_audio=cropped_audio_sample,
            output_directory=samples_output_directory,
        )
