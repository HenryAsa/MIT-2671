from generate_audio import generate_single_frequency



if __name__ == "__main__":
    frequency = 440
    sample_duration = 5

    for sample_rate in [1000, 44100, 88200, 96000, 192000]:
        for bit_depth in [8, 16, 24]:
            generate_single_frequency(frequency, sample_duration, sample_rate, bit_depth=bit_depth)
            generate_single_frequency(frequency, sample_duration, sample_rate, bit_depth=bit_depth)
            generate_single_frequency(frequency, sample_duration, sample_rate, bit_depth=bit_depth)
