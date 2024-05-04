from audio_frequency_generation import generate_single_frequency_audio_files
from constants import DATA_DIRECTORY
from datetime import datetime



if __name__ == "__main__":
    initial_time = datetime.now().strftime("%m-%d_%H-%M")
    output_directory = f'{DATA_DIRECTORY}/frequencies/{initial_time}'

    frequency = 440
    sample_duration = 5

    generate_single_frequency_audio_files(
        sample_rates=[1000, 2000, 4000, 8000, 44100, 88200, 96000, 192000],
        output_directory=output_directory,
    )
