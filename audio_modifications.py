import os
from pathlib import Path
import re
from pydub import AudioSegment
import numpy as np
import soundfile as sf

from constants import DATA_NORMALIZED_SAMPLES_DIRECTORY, DATA_RECORDED_SAMPLES_DIRECTORY, MP3_BITRATES, TEST_BIT_DEPTHS, TEST_SAMPLE_RATES
from audio_reading import simultaneous_record_playback
from utils import get_audio_params_from_filepath, initialize_data_folders



def crop_audio(
        input_file: str,
        start_ms: int,
        end_ms: int,
        output_directory: str,
        output_filename: str
    ) -> AudioSegment:
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
    if file_format not in {'flac', 'mp3', 'wav', 'aif'}:
        raise ValueError("Unsupported file format: Only .flac and .wav files are supported.")

    # Load the audio file
    audio: AudioSegment = AudioSegment.from_file(input_file, format=file_format)

    cropped_audio: AudioSegment = audio[start_ms:end_ms]

    ## TODO: MIGHT REMOVE THIS - SEE https://github.com/jiaaro/pydub/blob/master/API.markdown#audiosegmentapply_gain_stereo FOR MORE DETAILS
    cropped_audio = cropped_audio.set_channels(1)
    #############################

    os.makedirs(output_directory, exist_ok=True)
    cropped_audio.export(f'{output_directory}/{output_filename}', format='wav' if file_format in {'flac', 'aif'} else file_format)

    return cropped_audio


def convert_aif_to_wav(aif_path: str, wav_path: str) -> None:
    """
    Convert an AIF file to a WAV file without losing quality.

    Parameters:
        aif_path (str): The file path for the input AIF file.
        wav_path (str): The file path where the output WAV file will be saved.

    Returns:
        None: The function outputs the WAV file at the specified path.
    """
    # Load the AIF file
    data, sample_rate = sf.read(aif_path)

    # Export audio to WAV format
    sf.write(wav_path, data, sample_rate, subtype='PCM_24')


def modify_audio_sample(
        audio_segment: AudioSegment,
        output_filename: str,
        output_file_format: str,
        output_sample_rate: int = False,
        export_to_mp3: bool = False,
        output_bit_depth: int = False,
    ) -> None:
    if output_filename.find(".") != -1:
        assert output_filename.split(".")[-1] == output_file_format.split(".")[-1], f'expected output_filename to have same format as output_file_format, but "{output_filename.split(".")[-1]}" is not the same as "{output_file_format.split(".")[-1]}"'

    if output_sample_rate is not False:
        audio_segment = audio_segment.set_frame_rate(output_sample_rate)

    if output_bit_depth is not False:
        if output_bit_depth == 8:
            audio_segment_bd = audio_segment.set_sample_width(int(output_bit_depth/8))
            audio_segment_bd.export(out_f=output_filename, format=output_file_format.replace(".", ""))
        else:
            data = audio_segment._data
            data_array = np.frombuffer(data, dtype=np.int32) # must use int32, this data is actually in 32bit fixed signed
            sf.write(file=output_filename, data=data_array, samplerate=output_sample_rate, subtype=f'PCM_{int(output_bit_depth)}')

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

    for sample_rate in TEST_SAMPLE_RATES:
        for bit_depth in TEST_BIT_DEPTHS:
            output_filename = f'{audio_name}_S{sample_rate}_B{bit_depth}.wav'
            modify_audio_sample(
                audio_segment=original_audio,
                output_filename=f'{sample_output_directory}/{output_filename}',
                output_file_format='.wav',
                output_sample_rate=sample_rate,
                output_bit_depth=bit_depth,
            )

        if make_mp3_files is True and sample_rate <= 48000:
            output_filename = f'{audio_name}_S{sample_rate}_BR.mp3'
            modify_audio_sample(
                audio_segment=original_audio,
                output_filename=f'{sample_output_directory}/{output_filename}',
                output_file_format='.mp3',
                output_sample_rate=sample_rate,
                export_to_mp3=make_mp3_files,
            )

def normalize_audio(files, target_rms=0.1):
    """
    Normalize the volume of multiple audio files to a target loudness level.

    :param files: List of file paths to the audio files.
    :param target_dbfs: Target loudness level in dBFS.
    """
    for file_path in files:
        # Load the audio file
        data, samplerate = sf.read(file_path)

        # Calculate current RMS of the audio
        current_rms = np.sqrt(np.mean(data**2))

        # Calculate the gain factor
        gain = target_rms / current_rms

        # Normalize the audio data
        normalized_data = data * gain

        # Save the normalized audio file
        normalized_file_path = f'{os.path.splitext(file_path)[0]}_NORMALIZED{os.path.splitext(file_path)[1]}'.replace(DATA_RECORDED_SAMPLES_DIRECTORY, DATA_NORMALIZED_SAMPLES_DIRECTORY)
        os.makedirs(normalized_file_path[:normalized_file_path.rfind("/")], exist_ok=True)
        sf.write(normalized_file_path, normalized_data, samplerate, subtype='PCM_24')
        print(f"Normalized {file_path} and saved to {normalized_file_path}")


def average_audio_files(
        file_list: list[str],
    ):

    audios = []
    lengths = []
    rms_values = []

    for file in file_list:
        file_format = file.split('.')[-1]
        # Load each file as an AudioSegment
        audio: AudioSegment = AudioSegment.from_file(file, format=file_format)
        audios.append(audio.set_channels(1))
        lengths.append(len(audio))
        rms_values.append(audio.rms)

    # Calculate the average RMS value
    average_rms = np.mean(rms_values)

    # Determine the minimum length to align audio clips
    min_length = min(lengths)

    # Trim all audio segments to the minimum length
    trimmed_audios = [audio[:min_length] for audio in audios]

    # Convert all trimmed audios to arrays
    arrays = [np.array(audio.get_array_of_samples()) for audio in trimmed_audios]

    # Calculate the average of these arrays
    avg_array = np.mean(arrays, axis=0)

    # Dynamically calculate the max possible value based on bit depth
    sample_width = audios[0].sample_width
    max_possible_val = 32767

    # # Calculate the RMS of the average and scale to match average RMS of inputs
    # avg_rms = np.sqrt(np.mean(np.square(avg_array)))
    # scaling_factor = average_rms / avg_rms if avg_rms != 0 else 1
    # avg_array = np.int16(avg_array * scaling_factor)

    # Normalize to prevent clipping
    max_val = np.max(np.abs(avg_array))
    scaling_factor = max_possible_val / max_val if max_val != 0 else 1
    avg_array = np.int16(avg_array * scaling_factor)

    # Export the averaged audio
    output_file = re.sub(r'_TRIAL\d+', '_AVG', file)
    sf.write(file=output_file, data=avg_array, samplerate=audios[0].frame_rate, subtype=f'PCM_24')
    # averaged_audio.export(output_file, format="wav")


if __name__ == "__main__":

    samples_output_directory, recorded_output_directory = initialize_data_folders()

    audio_samples = {
        "hotel_california_intro": {
            "start_ms": 500,
            "end_ms": 2000,
            "filepath": "music/1-Hotel California.flac",
        },
        "hotel_california_guitar_solo": {
            "start_ms": 328000,
            "end_ms": 332000,
            "filepath": "music/1-Hotel California.flac",
        }
    }

    #### MAKE AUDIO SAMPLES WITH DIFFERENT PARAMETERS ####
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
            make_mp3_files=True,
        )
    ######################################################

    NUM_TRIALS = 10

    audio_files = sorted(str(filename) for filename in Path(samples_output_directory).rglob('*.[wav mp3]*'))

    for trial_num in range(1, NUM_TRIALS + 1):
        for filename in audio_files:
            file_params = get_audio_params_from_filepath(filename)
            output_file_directory = f'{recorded_output_directory}/{file_params["song_simple_name"]}'
            output_filename = f'{file_params["output_filename"].split(".")[0]}{f"_TRIAL{trial_num}"}.wav'
            print(file_params["output_filename"])
            os.makedirs(output_file_directory, exist_ok=True)
            simultaneous_record_playback(
                input_filename=filename,
                output_directory=output_file_directory,
                output_filename=output_filename,
            )
