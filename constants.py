"""`constants`

Contains common constant value definitions that are used throughout
the code repository.  Serves as the "source of truth" for common
constant values
"""

DATA_DIRECTORY = "data"
"""Default directory to store all of the raw data files"""

MP3_BITRATES = [
        "32k",  # Generally acceptable only for speech
        "96k",  # Generally used for speech or low-quality streaming.
        "128k", # mid-range bitrate quality.
        "160k", # mid-range bitrate quality.
        "192k", # medium quality bitrate.
        "256k", # a commonly used high-quality bitrate.
        "320k", # highest bitrate supported by MP3 standard
    ]
"""Typical Bitrates for MP3 files, in the readable type for pydub"""

RECORDING_SAMPLE_RATE = 96000
"""Default sampling rate of 96 kHz for recorded samples"""
