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

DATA_NORMALIZED_SAMPLES_DIRECTORY = "normalized_samples"
"""Directory name for the normalized recorded audio samples"""

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
    4000,
    6000,
    8000,
    10000,
    12000,
    16000,
    20000,
    24000,
    28000,
    32000,
    36000,
    40000,
    44100,
    48000,
    # 88200,
    # 96000,
    # 192000
]
"""Sample Rates (in Hz) of the audio test samples"""

TEST_BIT_DEPTHS = [8, 16, 24]
"""Bit depths to use for testing (in bits)"""


MATPLOTLIB_DEFAULTS = { ## https://matplotlib.org/stable/api/matplotlib_configuration_api.html#matplotlib.RcParams
    "axes.labelsize": 30,
    "axes.titlesize": 40,
    "figure.figsize": [20, 12],
    "figure.subplot.left": 0.075,
    "figure.subplot.right": 1 - 0.025,
    "figure.subplot.bottom": 0.075,
    "figure.subplot.top": 1 - 0.025,
    "legend.fontsize": 20,
    "lines.markersize": 10,
    "lines.linewidth": 6,
    "savefig.dpi": 600,
    "xtick.labelsize": 24,
    "ytick.labelsize": 24,
}
"""Default styling presets for Matplotlib plots"""
