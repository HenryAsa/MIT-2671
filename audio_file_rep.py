from utils import get_audio_params_from_filepath

class AudioFile:
    """
    Represents an audio file, encapsulating its properties and 
    providing methods to access specific attributes based on the file
    type.

    Parameters
    ----------
    file_path : str
        The file path to the audio file.

    Attributes
    ----------
    file_path : str
        The path to the audio file.
    file_type : str
        The file type of the audio file (e.g., '.mp3', '.wav').
    song_simple_name : str
        A simplified name representation of the song.
    output_file_name : str
        The name of the output file.
    sample_rate : int
        The sample rate of the audio file in Hz.
    bit_depth : int, optional
        The bit depth of the audio file. Only applicable to '.flac'
        and '.wav' file types. None for '.mp3'.
    bitrate : int, optional
        The bitrate of the audio file in kilobits per second. Only 
        applicable to '.mp3' file type. None for '.flac' and '.wav'.

    Methods
    -------
    get_by_sample_rate_name()
        Returns a string representation highlighting the sample rate
        and, depending on the file type, the bit depth or bitrate.

    get_by_file_type_name()
        Returns a string representation including the file type,
        sample rate, and either bit depth or bitrate as appropriate.

    Notes
    -----
    The class uses a custom comparator (`__lt__`) to define a sorting
    order based on the file type, sample rate, and either bit depth or
    bitrate depending on the file type.  This allows instances of
    `AudioFile` to be directly compared and sorted.

    Raises
    ------
    TypeError
        If attempting to compare an `AudioFile` instance with an
        unsupported or unknown file type.

    See Also
    --------
    get_audio_params_from_filepath : Function used to extract audio
        file parameters from a file path.

    Examples
    --------
    >>> audio_file = AudioFile("/path/to/song.wav")
    >>> print(audio_file.get_by_sample_rate_name())
    WAV: Bit Depth = 24
    >>> print(audio_file.get_by_file_type_name())
    Sample Rate = 44100 Hz, Bit Depth = 24
    """
    def __init__(self, file_path) -> None:
        self.file_path: str = file_path
        file_params = get_audio_params_from_filepath(filepath=file_path)
        self.file_type: str = file_params["filetype"]
        self.song_simple_name: str = file_params["song_simple_name"]
        self.output_file_name: str = file_params["output_filename"]
        self.sample_rate: int = file_params["sample_rate"]

        self.bit_depth: int | None = file_params["bit_depth"] if self.file_type in {'.flac', '.wav'} else None
        self.bitrate: int | None = file_params["bitrate"] if self.file_type == '.mp3' else None

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
    
    def __lt__(self, other):
        """
        Defines the less-than comparison between two `AudioFile`
        instances based on file type priority, sample rate, and bit
        depth or bitrate.

        The comparison logic is as follows:
        1. `.wav` files are considered greater than `.mp3` files.
        2. If file types are the same, instances are compared based on 
        `sample_rate`.
        3. If `sample_rate` is also equal, comparison is based on
        `bit_depth` for `.wav` and `.flac` files, and on `bitrate` for
        `.mp3` files.

        Parameters
        ----------
        other : AudioFile
            The other `AudioFile` instance to compare against.

        Returns
        -------
        bool
            True if `self` is considered less than `other`, False
            otherwise.

        Raises
        ------
        TypeError
            If attempting to compare an `AudioFile` instance with an
            unsupported or unknown file type, or if `other` is not an
            instance of `AudioFile`.

        Examples
        --------
        >>> file_a = AudioFile("/path/to/a.wav")  # Assume sample_rate=44100, bit_depth=24
        >>> file_b = AudioFile("/path/to/b.mp3")  # Assume sample_rate=44100, bitrate=320
        >>> file_a < file_b
        False

        >>> file_c = AudioFile("/path/to/c.wav")  # Assume sample_rate=48000, bit_depth=24
        >>> file_a < file_c
        True

        Note
        ----
        This method enables the use of sorting functions like
        `sorted()` on lists containing `AudioFile` instances, allowing
        them to be ordered according to the specified criteria.
        """
        # First, compare file types (.wav > .mp3)
        if self.file_type == other.file_type:
            # If file types are the same, sort by sample_rate
            if self.sample_rate == other.sample_rate:
                # If sample_rate is also equal, sort by bit_depth or bitrate depending on filetype
                if self.file_type in {'.flac', '.wav'}:
                    return self.bit_depth < other.bit_depth
                elif self.file_type == '.mp3':
                    return self.bitrate < other.bitrate
                else:
                    # For unknown or unsupported file types, consider them equal
                    raise TypeError(f'{self} is not comparable to {other}')
            else:
                return self.sample_rate < other.sample_rate
        else:
            return self.file_type < other.file_type

    def __repr__(self):
        return f"{self.song_simple_name} ({self.file_type}): SR={self.sample_rate}, {'BD=' + str(self.bit_depth) if self.bit_depth else 'BR=' + str(self.bitrate)}"
