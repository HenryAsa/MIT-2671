"""`constants`

Contains common constant value definitions that are used throughout
the code repository.  Serves as the "source of truth" for common
constant values
"""

DATA_AUDIO_SAMPLES_DIRECTORY = "audio_test_samples"
"""Directory name for audio samples (masters) that are being recorded
and played"""

DATA_DIRECTORY = "data"
"""Default directory to store all of the raw data files"""

DATA_RECORDED_SAMPLES_DIRECTORY = "recorded_samples"
"""Directory name for the recorded samples"""

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

RECORDED_BIT_DEPTH = 32
"""Default bit depth that the recordings were taken at"""

RECORDING_SAMPLE_RATE = 96000
"""Default sampling rate of 96 kHz for recorded samples"""

RECORDED_SAMPLE_FILENAME_PREFIX = "result_"
"""All recorded files should start with this prefix"""

TEST_SAMPLE_RATES = [
        1000,
        2000,
        4000,
        8000,
        16000,
        24000,
        32000,
        44100,
        48000,
        82000,
        96000,
        192000
    ]
"""Sample Rates (in Hz) of the audio test samples"""

TEST_BIT_DEPTHS = [8, 16, 24]
"""Bit depths to use for testing (in bits)"""
