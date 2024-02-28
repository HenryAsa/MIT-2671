## SOURCE: https://gist.github.com/nhrade/7cfcdf9e00602e52f516f90315b5d26e
## SOURCE: https://medium.com/@noahhradek/sound-synthesis-in-python-4e60614010da

from random import sample
import numpy as np
from scipy.io.wavfile import write, read
from scipy import signal, interpolate
from scipy.fft import fft, fftfreq
from scipy import interpolate
import matplotlib.pyplot as plt
from pydub import AudioSegment, playback
import wavio

DEFAULT_SAMPLE_RATE = 44100.0
DEFAULT_BITRATE = 8

"""
Signal class responsible for all signal processing
"""
class Signal:

    def __init__(self, time_length: float, ys: np.ndarray, sample_rate: int = DEFAULT_SAMPLE_RATE, bitrate: int = 16):
        """Initialize a Signal object to be used in the analyses later
        on throughout the sound tools

        Parameters
        ----------
        time_length : float
            float corresponding to the number of seconds the sample
            should be
        ys : np.array
            array of amplitude values corresponding to the amplitude
            of the desired wave
        sample_rate : int, optional
            sample rate of the audio file in Hz, by default DEFAULT_SAMPLE_RATE
        bitrate : int
            Bitrate of the sample (usually 8, 16, or 24)
        """
        self.time_length = time_length
        self.sample_rate = sample_rate
        self.ts = np.linspace(0, time_length, time_length * sample_rate, dtype=np.float32)
        self.ys = ys.astype(np.float32)
        self.bitrate = bitrate

    # add two signals of the same size
    def __add__(self, other):
        if self.ys.shape[0] != other.ys.shape[0]:
            raise ValueError(
                f"Dimension Mismatch: {self.ys.shape[0]} != {other.ys.shape[0]}"
            )
        return Signal(self.time_length, self.ys + other.ys, self.sample_rate, self.bitrate)

    # subtract two signals of the same size
    def __sub__(self, other):
        if self.ys.shape[0] != other.ys.shape[0]:
            raise ValueError(
                f"Dimension Mismatch: {self.ys.shape[0]} != {other.ys.shape[0]}"
            )
        return Signal(self.ts, self.ys - other.ys)

    # multiply by constant or by eleentwise signal
    def __mul__(self, other):
        # check if other is a number
        if isinstance(other, (int, float)):
            return Signal(self.time_length, other * self.ys, self.sample_rate, self.bitrate)
        elif isinstance(other, np.ndarray):
            if self.ys.shape[0] != other.ys.shape[0]:
                raise ValueError(
                    f"Dimension Mismatch: {self.ys.shape[0]} != {other.ys.shape[0]}"
                )
            return Signal(self.time_length, np.multiply(self.ys, other), self.sample_rate, self.bitrate)
        return None
    
    def __rmul__(self, other):
        return self.__mul__(other)

    # divide by factor or by elementwise signal
    def __truediv__(self, other):
        # check if other is a number
        if isinstance(other, (int, float)):
            return Signal(self.ts, self.ys / other)
        elif isinstance(other, np.ndarray):
            if self.ys.shape[0] != other.ys.shape[0]:
                raise ValueError(
                    f"Dimension Mismatch: {self.ys.shape[0]} != {other.ys.shape[0]}"
                )
            return Signal(self.ts, np.divide(self.ys, other), self.sample_rate, self.bitrate)
        return None

    # repeat signal n times
    # n - int number of repeats
    def repeat(self, n):
        # length = self.ts.shape[0]
        # ts = np.linspace(0, n * length, n * length * self.sample_rate, dtype=np.float32)
        ys = np.repeat(self.ys, n)
        return Signal(n * self.time_length, ys, self.sample_rate, self.bitrate)

    # play sound using pydub
    def play(self):
        aseg = AudioSegment(
            self.ys.tobytes(),
            frame_rate=self.sample_rate,
            sample_width=self.ys.dtype.itemsize,
            channels=1
        )
        playback.play(aseg)

    # save as wave file
    def to_wav(self, filename, directory: str = ""):
        wavio.write(f'{directory}/{filename}_S{self.sample_rate}_B{self.bitrate}.wav', self.ys, self.sample_rate, sampwidth=int(self.bitrate / 8))
        # write(filename, self.sample_rate, self.ys)

    # read from wave file
    # filename - string name of file
    @staticmethod
    def from_wav(filename):
        sample_rate, data = read(filename)
        length = data.shape[0] / sample_rate
        ys = data[:, 0] / np.max(np.abs(data[:, 0]), axis=0) # left channel
        ts = np.linspace(0., length, data.shape[0])
        return Signal(length, ys, sample_rate=sample_rate)  ## TODO: ENCODE BITRATE DATA

    # calculates the fft of the signal
    def fft(self):
        n = self.ys.shape[0]
        t = 1. / self.sample_rate
        # calculate fft amplitudes
        yf = fft(self.ys, n, )
        # calculate fft frequencies
        xf = fftfreq(n, t)[:n//2]
        return xf, yf

    # plot frequencies in range rng
    # rng - tuple range of frequencies
    def plot_fft(self, rng=(0, 2000)):
        xf, yf = self.fft()
        n = self.ys.shape[0]
        plt.plot(xf, 2.0/n * np.abs(yf[:n//2]))
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Amplitude")
        plt.title("FFT")
        plt.xlim(*rng)
        plt.grid()
        plt.show()

    # plot with num_samples, discrete shows only discrete signal
    # num_samples - int number of samples to plot
    # discrete - bool whether to plot on discrete scale
    def plot(self, num_seconds, discrete=False):
        num_samples = min(round(num_seconds * self.sample_rate), len(self.ts))
        if discrete:
            plt.scatter(self.ts[:num_samples], self.ys[:num_samples])
        else:
            plt.plot(self.ts[:num_samples], self.ys[:num_samples])
        plt.show()

    def plot_waveform(self, num_points: int):
        assert num_points <= len(self.ys), f'Attempting to plot more points of the waveform than the waveform has.  Desired Plotting Range: {num_points}.  Max Plotting Range: {len(self.ys)}'

        plt.step(self.ts[:num_points], self.ys[:num_points], where='mid', label=f'{self.bitrate} bit, {self.sample_rate} Hz')
        # plt.plot(time_period, data_period, 'o--', label=f'Sample Rate = {sample_rate}', color='grey', alpha=0.3)
        # plt.plot(time_period, data_period, label=f'Sample Rate = {sample_rate}')

        # plt.title(f'Waveform of Recordings with Different Sample and Bit Rates for f = {frequency} Hz')
        plt.legend()
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")
        # plt.show()

    # plot spectrogram image
    def plot_spectrogram(self):
        f, t, Sxx = signal.spectrogram(self.ys, self.sample_rate)
        plt.pcolormesh(t, f, Sxx, shading='gouraud')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.show()

    # get size of signal in number of samples
    def size(self):
        return self.ys.shape[0]

    # get length of sample in seconds
    def length(self):
        return self.ys.shape[0] / self.sample_rate

    # get resampled signal from start to end
    def get(self, start, end):
        size = abs(end - start)
        ts = np.linspace(0, size, size * self.sample_rate, dtype=np.float32)
        return Signal(size, self.ys[start:end+1], self.sample_rate, self.bitrate)

    # filter below or above cutoff
    # cutoff - int cutoff frequency
    # ftype - type of filter
    # order of filter, higher means faster cutoff
    def filter(self, cutoff, ftype="lowpass", order=5):
        sos = signal.butter(order, cutoff, fs=self.sample_rate, btype=ftype, output="sos")
        fy = signal.sosfilt(sos, self.ys)
        return Signal(self.time_length, fy, sample_rate=self.sample_rate, bitrate=self.bitrate)

    def __str__(self):
        return f'Signal[Length: {self.time_length} sec, SampleRate: {self.sample_rate} Hz, BitRate: {self.bitrate} bit]'

class Sine(Signal):
    def __init__(self, frequency: float, time_length: float, sample_rate: int, bitrate: int) -> None:
        """Sine Signal Waveform

        Parameters
        ----------
        frequency : float
            Frequency of the sine wave (in Hz)
        time_length : float
            Length of time (in seconds) of the waveform
        sample_rate : int
            Sample rate (in Hz) of the sine wave
        bitrate : int
            Bitrate of the file to encode
        """
        self.frequency = frequency
        self.time_length = time_length
        self.ts = np.linspace(0, time_length, time_length * sample_rate, dtype=np.float32)
        self.ys = np.sin(2 * np.pi * frequency * self.ts)
        self.sample_rate = sample_rate
        self.bitrate = bitrate

        super().__init__(
            time_length = time_length,
            ys = self.ys,
            sample_rate = sample_rate,
            bitrate = bitrate,
        )

    def plot_waveform(self, num_waves: float) -> None:
        num_points = round(num_waves * self.sample_rate / self.frequency)
        assert num_points <= len(self.ys), f'Desired number of waves to plot exceeds the number of waves in the sample.  Desired number of waves: {num_waves}.  Number of Waves in Waveform: {round(len(self.ys) / self.frequency)}'
        super().plot_waveform(num_points=num_points)

    def __str__(self):
        return f'Sine[Length: {self.time_length} sec, SampleRate: {self.sample_rate} Hz, BitRate: {self.bitrate} bit, Freq: {self.frequency} Hz]'



class Square(Signal):
    def __init__(self, frequency, amp=1., length=1):
        self.frequency = frequency
        self.ts = np.linspace(0, length, length * DEFAULT_SAMPLE_RATE, dtype=np.float32)
        self.ys = amp * signal.square(2 * np.pi * frequency * self.ts)
        super().__init__(self.time_length, self.ys)

class Sawtooth(Signal):
    def __init__(self, freq, amp=1., length=1):
        self.ts = np.linspace(0, length, length * DEFAULT_SAMPLE_RATE, dtype=np.float32)
        self.ys = amp * signal.sawtooth(2 * np.pi * freq * self.ts)
        super().__init__(self.time_length, self.ys)


class Chirp(Signal):
    def __init__(self, f0, t1, f1, amp=1., length=1):
        self.ts = np.linspace(0, length, length * DEFAULT_SAMPLE_RATE, dtype=np.float32)
        self.ys = amp * signal.chirp(self.ts, f0, t1, f1)
        super().__init__(self.time_length, self.ys)
