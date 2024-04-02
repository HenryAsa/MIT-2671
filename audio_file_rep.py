from utils import get_audio_params_from_filepath

class AudioFile:
    def __init__(self, file_path) -> None:
        self.file_path = file_path
        file_params = get_audio_params_from_filepath(filepath=file_path)
        self.file_type = file_params["filetype"]
        self.song_simple_name = file_params["song_simple_name"]
        self.output_file_name = file_params["output_filename"]
        self.sample_rate = file_params["sample_rate"]

        self.bit_depth = file_params["bit_depth"] if self.file_type in {'.flac', '.wav'} else None
        self.bitrate = file_params["bitrate"] if self.file_type == '.mp3' else None

    # @property
    # def bitrate(self):
    #     assert self.file_type == '.mp3', f'bitrate is only attainable for .mp3 files, but this file is of type {self.file_type}'
    #     return self.bitrate

    # @property
    # def bit_depth(self):
    #     assert self.file_type in {'.flac', '.wav'}, f'bit depth is only attainable for .flac and .wav files, but this file is of type {self.file_type}'
        # return self.bit_depth

    def get_by_sample_rate_name(self):
        if self.file_type == '.mp3':
            return f'MP3: BitRate = {self.bitrate} kb/s'
        else:
            return f'WAV: Bit Depth = {self.bit_depth}'

    def get_by_file_type_name(self):
        if self.file_type == '.mp3':
            return f'Sample Rate = {self.sample_rate} Hz, BitRate = {self.bitrate} kb/s'
        else:
            return f'Sample Rate = {self.sample_rate} Hz, Bit Depth = {self.bit_depth}'
