import math
import time
import numpy as np
import wavio
import matplotlib.pyplot as plt
import os
import wave
import sounddevice as sd
from pydub import AudioSegment
from pydub.playback import play
from pathlib import Path
import threading
import queue
import soundfile as sf
import sys



from signals import Signal, Sine

folder_name = "generated_audio"


frequency = 440.0                       # sound frequency (Hz)
sample_duration = 5                     # sample duration (seconds)

def generate_single_frequency(frequency, sample_duration, sample_rate, sample_bit_rate, folder_name):
    sample_number = int(sample_rate * sample_duration)          # number of samples
    time_vals = np.arange(sample_number)/sample_rate            # grid of time values
    x = np.sin(2*np.pi * frequency * time_vals)
    wavio.write(f'{folder_name}/sine_F{frequency}_S{sample_rate}_B{sample_bit_rate}.wav', x, sample_rate, sampwidth=int(sample_bit_rate/8))



def generate_audio_files():
    frequencies = [440, 1000, 2000, 4000, 8000, 12000, 16000]
    sample_rates = [1000, 2000, 4000, 8000, 16000, 44100, 82000, 96000, 192000]
    bitrates = [8, 16, 24]
    sample_duration = 5

    for freq in frequencies:
        folder_name = f'F{freq}'
        for sample_rate in sample_rates:
            for bitrate in bitrates:
                if sample_rate <= 2*freq:
                    ## ALIASING IS OCCURING
                    continue
                os.makedirs(f'audio_test_samples/{folder_name}', exist_ok=True)
                generate_single_frequency(freq, sample_duration, sample_rate, bitrate, f'audio_test_samples/{folder_name}')
                print(f'sine_F{frequency}_S{sample_rate}_B{bitrate}.wav')


# q = queue.Queue()

# def callback(indata, frames, time, status):
#     """This is called for each audio block from the microphone"""
#     if status:
#         print(status)
#     q.put(indata.copy())

# def record_audio(duration, samplerate):
#     """Record audio from the microphone"""
#     with sd.InputStream(samplerate=samplerate, channels=1, callback=callback, dtype='float32'):
#         sd.sleep(int(duration * 1000))

# def play_audio(filename):
#     """Play an audio file"""
#     # Open the WAV file
#     wf = wave.open(filename, 'rb')
#     # Get audio file properties
#     samplerate = wf.getframerate()
#     channels = wf.getnchannels()
#     # Start playback
#     stream = sd.OutputStream(samplerate=samplerate, channels=channels)
#     stream.start()
#     data = wf.readframes(1024)
#     while len(data) > 0:
#         stream.write(data)
#         data = wf.readframes(1024)
#     stream.stop()
#     stream.close()
#     wf.close()


q = queue.Queue()


def callback_in(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    if status:
        print(status, file=sys.stderr)
    q.put(indata.copy())


def record_audio_samples():

    RECORDING_SAMPLE_RATE = 96000

    print(f'INPUT SETTINGS: \t{sd.check_input_settings()}')
    print(f'OUTPUT SETTINGS:\t{sd.check_output_settings()}')

    audio_files = list(Path("audio_test_samples").rglob("*.wav"))
    output_folder = f'recorded_samples'
    os.makedirs(output_folder, exist_ok=True)

    for filename in audio_files:
        # read MP3 file
        file_params = str(filename).split("_")
        filetype = file_params[2].split("/")[-1]
        freq = file_params[3][1:]
        sample_rate = file_params[4][1:]
        bit_depth = file_params[-1].split(".")[0][1:]
        
        output_filename = f'result_{filetype}_F{freq}_S{sample_rate}_B{bit_depth}.wav'
        os.makedirs(f'{output_folder}/F{freq}', exist_ok=True)
        print(filename)
        song: AudioSegment = AudioSegment.from_wav(filename)
        # print(f'{song.sample_width}  {song.frame_rate}  {song.frame_count()}  {song.frame_width}')
        # song = AudioSegment.from_wav("audio_file.wav")
        # you can also read from other formats such as MP4
        # song = AudioSegment.from_file("audio_file.mp4", "mp4")
        # play(song)
        

        # Extract the samplerate from the audio file
        filename = str(filename)
        # filename = "04 - David Elias - The Window - Vision of Her (16b-44.1).wav"
        # filename = "short_song.wav"
        # output_filename = "testA.wav"
        # wf = wave.open(filename, 'rb')
        # samplerate = wf.getframerate()
        # wf.close()

        # # Start recording and playback threads
        # record_thread = threading.Thread(target=record_audio, args=(6, 192000))
        # play_thread = threading.Thread(target=play_audio, args=(filename,))

        # record_thread.start()
        # play_thread.start()

        # # Wait for playback to finish
        # play_thread.join()
        # # Wait for recording to finish
        # record_thread.join()

        # # Save the recorded audio to a WAV file
        # with wave.open(f'{output_folder}/F{freq}/{output_filename}', 'wb') as wave_file:
        #     wave_file.setnchannels(1)
        #     wave_file.setsampwidth(3)
        #     wave_file.setframerate(192000)
        #     while not q.empty():
        #         frame = q.get()
        #         wave_file.writeframes(frame)
        sound_data, _fs = sf.read(filename)
        print(len(sound_data) / _fs)
        print(_fs)

        # recorded_audio = sd.playrec(
        #     data=sound_data,
        #     samplerate=_fs,
        #     # data=np.array(song.get_array_of_samples()).astype(np.float64),
        #     # samplerate=int(sample_rate),
        #     channels=1,
        #     blocking=True,
        # )
        



        # # recorded_audio = np.empty(shape=math.ceil(RECORDING_SAMPLE_RATE * (len(sound_data)/_fs)))
        # print("ABOUT TO START RECORDING")
        # # recorded_audio = sd.rec(out=recorded_audio, samplerate=RECORDING_SAMPLE_RATE, channels=1, dtype="float64")
        # recording_frames = math.ceil(RECORDING_SAMPLE_RATE * (len(sound_data)/_fs))
        # recorded_audio = sd.rec(frames=recording_frames, samplerate=RECORDING_SAMPLE_RATE, channels=1, dtype="float64")
        # print("ABOUT TO START PLAYING")
        # # sd.play(
        # #     data = sound_data,
        # #     samplerate=_fs,
        # #     blocking=True,
        # # )
        # print("FINISHED PLAYING")
        # print(recorded_audio)

        # # audio = AudioSegment(recorded_audio)
        # wavio.write(f'{output_folder}/F{freq}/{output_filename}', recorded_audio, RECORDING_SAMPLE_RATE, sampwidth=4)




        # # print(audio.get_array_of_samples())

        # # with sf.SoundFile(args.filename, mode='x', samplerate=args.samplerate,
        # #               channels=args.channels, subtype=args.subtype) as file:
        # recording = sd.InputStream(samplerate=RECORDING_SAMPLE_RATE,
        #                     channels=1, callback=callback_in)
        #     # print('#' * 80)
        #     # print('press Ctrl+C to stop the recording')
        #     # print('#' * 80)
        # print("ABOUT TO START PLAYING")
        # sd.play(
        #     data = sound_data,
        #     samplerate=_fs,
        #     blocking=True,
        # )
        # # recorded_audio = q.get()
        # sd.stop()
        # print(recording.read(recording.read_available))
        # recording.abort()
        # wavio.write(f'{output_folder}/F{freq}/{output_filename}', recording, RECORDING_SAMPLE_RATE, sampwidth=4)



        # print(audio.get_array_of_samples())
        duration = len(sound_data)/_fs

        with sf.SoundFile(f'{output_folder}/F{freq}/{output_filename}', mode='w+', samplerate=RECORDING_SAMPLE_RATE,
                      channels=1, subtype='PCM_32', format='WAV') as file:
            with sd.InputStream(samplerate=RECORDING_SAMPLE_RATE, channels=1, callback=callback_in):
                # print('#' * 80)
                # print('press Ctrl+C to stop the recording')
                # print('#' * 80)
                print("ABOUT TO START PLAYING")
                sd.play(data = sound_data, samplerate=_fs, blocking=False)
                start_time = time.time()
                while time.time() - start_time <= duration:
                    file.write(q.get())
        

                
        # recorded_audio = q.get()
        sd.stop()
        raise TypeError


    raise TypeError
    fs=19200
    duration = 10  # seconds
    myrecording = sd.rec(duration * fs, samplerate=fs, channels=2, dtype='float64')
    print("Recording Audio for %s seconds" %(duration))
    sd.wait()
    print("Audio recording complete , Playing recorded Audio")
    sd.play(myrecording, fs)
    sd.wait()
    print("Play Audio Complete")

if __name__ == "__main__":
    record_audio_samples()
    generate_audio_files()
    # frequency = 440
    # sample_duration = 5

    # test_signal_1000_8 = Sine(frequency=frequency, time_length=sample_duration, sample_rate=1000, bitrate=8)
    # test_signal_44100_8 = Sine(frequency=frequency, time_length=sample_duration, sample_rate=44100, bitrate=8)
    # test_signal_44100_16 = Sine(frequency=frequency, time_length=sample_duration, sample_rate=44100, bitrate=16)

    # test_signal_1000_8_audio = Sine.from_wav(f'{folder_name}/sine_F{frequency}_S1000_B8.wav', bitrate=8, frequency=frequency)
    # test_signal_1000_8_discrete = Sine(frequency=frequency, time_length=sample_duration, sample_rate=1000, bitrate=8, discretize = True)


    # num_waves_to_plot = 5

    # test_signal_1000_8.plot_waveform(num_waves=num_waves_to_plot)
    # test_signal_1000_8_audio.plot_waveform(num_waves=num_waves_to_plot)
    # test_signal_1000_8_discrete.plot_waveform(num_waves=num_waves_to_plot)
    # # test_signal_44100_8.plot_waveform(num_waves=num_waves_to_plot)
    # # test_signal_44100_16.plot_waveform(num_waves=num_waves_to_plot)

    # plt.show()
    raise TypeError
