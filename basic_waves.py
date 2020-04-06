# -*- coding: utf-8 -*-
# @Author: ZMJ
# @Date:   2020-03-28 18:26:34
# @Last Modified by:   ZMJ
# @Last Modified time: 2020-04-06 17:32:00
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read as wave_reader
from scipy.io.wavfile import write as wave_writer
import pyaudio
class Signal(object):
	"""docstring for Signal"""
	def __init__(self, framerate):
		super(Signal, self).__init__()
		self.framerate = framerate

	def set_values(self, x_coords, y_coords):
		self.x_coords = x_coords
		self.y_coords = y_coords
		self.y_coords = self.y_coords.astype(np.float32)

	def get_frequencies(self, window_name = "none"):
		"""[summary]
		
		[description]
		
		Arguments:
			fs {[type]} -- [the sampling frequency for fft]
			window_name {[type]} -- [there are some windows that already been implemented in numpy, e.g. hamming, bartlett, blackman, hanning, and kaiser]

		"""
		fft_size = self.framerate
		freqs = np.linspace(0, fft_size, len(self.x_coords))
		if window_name == "none":
			window = np.ones_like(self.y_coords)
		elif window_name == "hamming":
			window = np.hamming(len(self.y_coords))
		elif window_name == "hanning":
			window = np.hanning(len(self.y_coords))
		elif window_name == "kaiser":
			window = np.kaiser(len(self.y_coords, 0))##beta = 0 means a square window
		elif window_name == "bartlett":
			window = np.bartlett(len(y_coords))
		elif window_name == "blackman":
			window = np.blackman(len(y_coords))
		amps = np.abs(np.fft.fft(window*self.y_coords))
		##normalization
		# amps /= len(self.x_coords)
		##half
		half = len(self.x_coords)//2
		freqs = freqs[:half]
		amps = amps[:half]
		return freqs, amps

	def write_wave(self, save_path):
		double_channels =  np.zeros((len(self.y_coords), 2))
		double_channels[:,0] = self.y_coords
		double_channels[:,1] = self.y_coords
		wave_writer(save_path, self.framerate, double_channels)

	def play_wave(self):
		#instantiate PyAudio  
		p = pyaudio.PyAudio()  
		#open stream  
		stream = p.open(format = pyaudio.paFloat32,  
		                channels = 1,  
		                rate = self.framerate,  
		                output = True) 
		#play stream  
		stream.write(self.y_coords) 
		#stop stream  
		stream.stop_stream()  
		stream.close()  
		#close PyAudio  
		p.terminate()		

class Wave(Signal):
	"""docstring for ClassName"""
	def __init__(self, period, framerate):
		super(Wave, self).__init__(framerate)
		self.period = period

	def segment(self, start=0, duration=5):
		"""[summary]
		
		[description]
		
		Keyword Arguments:
			start {number} -- [the required start point, located between 0 and 1] (default: {0})
			duration {number} -- [the number of period] (default: {5})
		"""

		min_tick = 1 / self.framerate
		start_pnt = int(start / min_tick)
		end_pnt = start_pnt + int(self.period*duration / min_tick)
		wave = Wave(self.period, self.framerate)	
		wave.set_values(self.x_coords[start_pnt:end_pnt], self.y_coords[start_pnt:end_pnt])
		return wave

	def __add__(self, wave):
		new_y_coords = self.y_coords + wave.y_coords
		signal = Signal(self.framerate)
		signal.set_values(self.x_coords, new_y_coords)
		return signal

	def get_GCD(self, number1, number2):
		"""[summary]
		return the greatest common division of number1 and number2
		[description]
		
		Arguments:
			number1 {[type]} -- [description]
			number2 {[type]} -- [description]
		"""
		gcd = 2
		while True:
			if number1 % gcd == 0 and number2 % gcd == 0:
				break
			gcd += 1
		return gcd

class SineWave(Wave):
	"""docstring for SineWave"""
	def __init__(self, freq = 100, amp = 1., offset = 0., framerate = 11025, duration = 1):
		"""[summary]
		return a sine wave pnts in 1 second
		[description]
		freq: the reauired frequency
		amp: the required amplitude
		offset: the reqiured phase difference (angle, in radians)
		framerate: the required framerate (FPS), representing the number of samplings
		"""
		super(SineWave, self).__init__(1/freq, framerate)
		self.amp = amp
		self.offset = offset
		self.freq = freq

		self.x_coords = np.linspace(0, duration, duration*framerate)
		self.w = 2 * np.pi * freq
		self.y_coords = np.array([amp * np.sin(self.w * x + offset) for x in self.x_coords])
		self.y_coords=self.y_coords.astype(np.float32)

class TriangleWave(Wave):
	"""docstring for TriangleWave"""
	def __init__(self, freq = 100, amp = 1, offset = 0, framerate = 11025, duration = 1):
		super(TriangleWave, self).__init__(1/freq, framerate)
		self.amp = amp
		self.offset = offset
		self.freq = freq	

		self.x_coords = np.linspace(0, duration, duration*framerate)
		cycles = self.freq * self.x_coords + self.offset/(2*np.pi)
		fracs, _ = np.modf(cycles)
		self.y_coords =  (fracs - 0.5)/0.5*amp	

class SquareWave(Wave):
	"""docstring for SquareWave"""
	def __init__(self, freq = 100, amp = 1, offset = 0, framerate = 11025, duration = 1):
		super(SquareWave, self).__init__(1/freq, framerate)
		self.amp = amp
		self.offset = offset
		self.freq = freq	

		self.x_coords = np.linspace(0, duration, duration*framerate)
		cycles = self.freq * self.x_coords + self.offset/(2*np.pi)
		fracs, _ = np.modf(cycles)
		self.y_coords =  np.sign(fracs - 0.5)*self.amp

class Chirp(Signal):
	"""docstring for Chirp"""
	def __init__(self, start_freq = 220, end_freq = 440, amp = 1, offset = 0, framerate = 11025, duration = 1):
		super(Chirp, self).__init__(framerate)
		self.amp = amp
		self.offset = offset

		##generate chirp signal 
		self.x_coords = np.linspace(0, duration, duration*framerate)
		freqs = np.linspace(start_freq, end_freq, duration*framerate-1)
		dts = np.diff(self.x_coords)
		dphis = 2*np.pi*freqs*dts+offset
		phases = np.cumsum(dphis)
		phases = np.insert(phases, 0, 0)
		self.y_coords = self.amp*np.sin(phases)

class ExpoChirp(Signal):
	"""docstring for ExpoChirp"""
	def __init__(self, start_freq = 220, end_freq = 440, amp = 1, offset = 0, framerate = 11025, duration = 1):
		super(ExpoChirp, self).__init__(framerate)
		self.amp = amp
		self.offset = offset

		##generate exponential chirp signal 
		self.x_coords = np.linspace(0, duration, duration*framerate)
		freqs = np.logspace(np.log10(start_freq), np.log10(end_freq), duration*framerate-1)
		dts = np.diff(self.x_coords)
		dphis = 2*np.pi*freqs*dts+offset
		phases = np.cumsum(dphis)
		phases = np.insert(phases, 0, 0)
		self.y_coords = self.amp*np.sin(phases)
		
				
if __name__ == '__main__':
	wave1 = SineWave(freq = 20).segment(duration = 3.2)
	wave2 = SquareWave(freq = 20).segment(duration = 3.7)
	wave3 = TriangleWave(freq = 20).segment(duration = 3.4)
	wave4 = Chirp(start_freq = 20, end_freq = 40)

	plt.subplot(3, 4, 1)
	plt.plot(wave1.x_coords, wave1.y_coords)
	plt.subplot(3, 4, 2)
	plt.plot(wave2.x_coords, wave2.y_coords)
	plt.subplot(3, 4, 3)
	plt.plot(wave3.x_coords, wave3.y_coords)
	plt.subplot(3, 4, 4)
	plt.plot(wave4.x_coords, wave4.y_coords)
	plt.subplot(3, 4, 5)
	freqs, amps = wave1.get_frequencies()
	plt.plot(freqs, amps)
	plt.xlim(0, 50)
	plt.subplot(3, 4, 6)
	freqs, amps = wave2.get_frequencies()
	plt.plot(freqs, amps)
	plt.xlim(0, 1000)
	plt.subplot(3, 4, 7)
	freqs, amps = wave3.get_frequencies()
	plt.plot(freqs, amps)
	plt.subplot(3, 4, 8)
	freqs, amps = wave4.get_frequencies()
	plt.plot(freqs, amps)
	plt.xlim(0, 1000)	

	plt.subplot(3, 4, 9)
	freqs, amps = wave1.get_frequencies(window_name = "hamming")
	plt.plot(freqs, amps)
	plt.xlim(0, 50)
	plt.subplot(3, 4, 10)
	freqs, amps = wave2.get_frequencies(window_name = "hamming")
	plt.plot(freqs, amps)
	plt.xlim(0, 1000)
	plt.subplot(3, 4, 11)
	freqs, amps = wave3.get_frequencies(window_name = "hamming")
	plt.plot(freqs, amps)
	plt.subplot(3, 4, 12)
	freqs, amps = wave4.get_frequencies(window_name = "hamming")
	plt.plot(freqs, amps)
	plt.xlim(0, 1000)	
	plt.show()

	# wave5 = Chirp(duration = 10)
	# wave5.play_wave()

	# wave6 = ExpoChirp(duration = 10)
	# wave6.play_wave()



