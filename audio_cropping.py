from pydub import AudioSegment
from constants import MP3_BITRATES
import numpy as np
import soundfile as sf



def crop_audio(input_file: str, start_ms: int, end_ms: int, output_filename: str) -> AudioSegment:
    """
    Crop the given audio file between the specified start and end
    times and saves it

    Parameters
    ----------
    input_file : str
        The path to the input .flac or .wav audio file
    start_ms : int
        The start time in milliseconds from which to start cropping
    end_ms : int
        The end time in milliseconds at which to stop cropping
    output_filename : str
        The path to the output .flac or .wav audio file
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
    if file_format not in {'mp3', 'flac', 'wav'}:
        raise ValueError("Unsupported file format: Only .flac and .wav files are supported.")

    # Load the audio file
    audio: AudioSegment = AudioSegment.from_file(input_file, format=file_format)

    cropped_audio: AudioSegment = audio[start_ms:end_ms]

    cropped_audio.export(output_filename, format=file_format)
    
    return cropped_audio

    # sf.write(
    #     file=f'{output_filename.split(".")[0]}_24.{file_format}',
    #     data=np.frombuffer(cropped_audio._data,
    #     dtype=np.int32).reshape(-1, cropped_audio.channels),
    #     samplerate=cropped_audio.frame_rate,
    #     subtype='PCM_32',
    # )

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
        audio_segment.export(out_f=output_filename, format=output_file_format, bitrate=int(output_bit_depth/8))

    if export_to_mp3 is not False:
        for mp3_bitrate in MP3_BITRATES:
            audio_segment.export(out_f=output_filename, format='mp3', bitrate=mp3_bitrate)


if __name__ == "__main__":

    cropped_audio = crop_audio("1-Hotel California.flac", 0, 9000, "music/CROPPED.wav")
