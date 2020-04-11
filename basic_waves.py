# -*- coding: utf-8 -*-
# @Author: ZMJ
# @Date:   2020-03-28 18:26:34
# @Last Modified by:   ZMJ
# @Last Modified time: 2020-04-11 17:15:52
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read as wave_reader
from scipy.io.wavfile import write as wave_writer
import pyaudio
class Wave(object):
	"""docstring for Wave"""
	def __init__(self, framerate):
		super(Wave, self).__init__()
		self.framerate = framerate

	def set_values(self, x_coords, y_coords):
		self.x_coords = x_coords
		self.y_coords = y_coords
		self.y_coords = self.y_coords.astype(np.float32)

	def get_frequencies(self, window_name = "none", full = False):
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
		##half
		if not full:
			half = len(self.x_coords)//2
			freqs = freqs[:half]
			amps = amps[:half]
		return freqs, amps

	def get_power(self, window_name = "none"):
		freqs, amps = self.get_frequencies(window_name)
		return freqs, amps*amps

	def get_cumsum_power(self, window_name = "none"):
		freqs, powers = self.get_power()
		cp = np.cumsum(powers)
		cp /= cp[-1]
		return freqs, cp

	def write_wave(self, save_path):
		double_channels =  np.zeros((len(self.y_coords), 2))
		double_channels[:,0] = self.y_coords
		double_channels[:,1] = self.y_coords
		wave_writer(save_path, self.framerate, double_channels)

	def __add__(self, wave):
		new_y_coords = self.y_coords + wave.y_coords
		wave = Wave(self.framerate)
		wave.set_values(self.x_coords, new_y_coords)
		return wave

	def compute_autocorr(self):
		lags = np.arange(len(self.y_coords)//2)
		corrs = [self.serial_autocorr(lag) for lag in lags]
		return lags, corrs

	def serial_autocorr(self, lag):
		length = len(self.y_coords)
		y1 = self.y_coords[lag:]
		y2 = self.y_coords[:length-lag]
		corr = np.corrcoef(y1, y2, ddof=0)
		return corr[0,1]

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

	def discrete_cosine_synthesize_sumption(self, amps, fs):
		"""[summary] discrete cosine synthesize by simply sumption of multiple cosine signals
		
		[description]
		
		Arguments:
			amps {[type]} -- [description] series of amptitudes
			fs {[type]} -- [description] series of frequencies
			ts {[type]} -- [description] series of timepoints
		"""
		wave = SineSignal(freq = fs[0], amp = amps[0], framerate = self.framerate, offset = np.pi/2)
		for f,a in zip(fs[1:], amps[1:]):
			w = SineSignal(freq = f, amp = a, framerate = self.framerate, offset = np.pi/2)
			wave = wave+w
		return wave

	def discrete_cosine_synthesize_linalg(self, amps, fs):
		"""[summary] discrete cosine synthesize by linear algebra
		
		[description]
		
		Arguments:
			amps {[type]} -- [description]
			fs {[type]} -- [description]
			framerate {[type]} -- [description]
		
		Returns:
			[type] -- [description]
		"""
		x_coords = np.linspace(0, 1, self.framerate)
		args = np.outer(x_coords, fs)
		M = np.cos(2*np.pi*args)
		y_coords = np.dot(M, amps)

		wave = Wave(self.framerate)
		wave.set_values(x_coords, y_coords)
		return wave

class Signal(Wave):
	"""docstring for ClassName"""
	def __init__(self, period, framerate):
		super(Signal, self).__init__(framerate)
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
		signal = Signal(self.period, self.framerate)	
		signal.set_values(self.x_coords[start_pnt:end_pnt], self.y_coords[start_pnt:end_pnt])
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

class SineSignal(Signal):
	"""docstring for SineSignal"""
	def __init__(self, freq = 100, amp = 1., offset = 0., framerate = 11025, duration = 1):
		"""[summary]
		return a sine wave pnts in 1 second
		[description]
		freq: the reauired frequency
		amp: the required amplitude
		offset: the reqiured phase difference (angle, in radians)
		framerate: the required framerate (FPS), representing the number of samplings
		"""
		super(SineSignal, self).__init__(1/freq, framerate)
		self.amp = amp
		self.offset = offset
		self.freq = freq

		self.x_coords = np.linspace(0, duration, duration*framerate)
		self.w = 2 * np.pi * freq
		self.y_coords = np.array([amp * np.sin(self.w * x + offset) for x in self.x_coords])
		self.y_coords=self.y_coords.astype(np.float32)

class TriangleSignal(Signal):
	"""docstring for TriangleSignal"""
	def __init__(self, freq = 100, amp = 1, offset = 0, framerate = 11025, duration = 1):
		super(TriangleSignal, self).__init__(1/freq, framerate)
		self.amp = amp
		self.offset = offset
		self.freq = freq	

		self.x_coords = np.linspace(0, duration, duration*framerate)
		cycles = self.freq * self.x_coords + self.offset/(2*np.pi)
		fracs, _ = np.modf(cycles)
		self.y_coords =  (fracs - 0.5)/0.5*amp	

class SquareSignal(Signal):
	"""docstring for SquareSignal"""
	def __init__(self, freq = 100, amp = 1, offset = 0, framerate = 11025, duration = 1):
		super(SquareSignal, self).__init__(1/freq, framerate)
		self.amp = amp
		self.offset = offset
		self.freq = freq	

		self.x_coords = np.linspace(0, duration, duration*framerate)
		cycles = self.freq * self.x_coords + self.offset/(2*np.pi)
		fracs, _ = np.modf(cycles)
		self.y_coords =  np.sign(fracs - 0.5)*self.amp

class Chirp(Wave):
	"""docstring for Chirp"""
	def __init__(self, start_freq = 220, end_freq = 440, amp = 1, offset = 0, framerate = 11025, duration = 1):
		super(Chirp, self).__init__(framerate)
		self.amp = amp
		self.offset = offset

		##generate chirp wave 
		self.x_coords = np.linspace(0, duration, duration*framerate)
		freqs = np.linspace(start_freq, end_freq, duration*framerate-1)
		dts = np.diff(self.x_coords)
		dphis = 2*np.pi*freqs*dts+offset
		phases = np.cumsum(dphis)
		phases = np.insert(phases, 0, 0)
		self.y_coords = self.amp*np.sin(phases)

class ExpoChirp(Wave):
	"""docstring for ExpoChirp"""
	def __init__(self, start_freq = 220, end_freq = 440, amp = 1, offset = 0, framerate = 11025, duration = 1):
		super(ExpoChirp, self).__init__(framerate)
		self.amp = amp
		self.offset = offset

		##generate exponential chirp wave 
		self.x_coords = np.linspace(0, duration, duration*framerate)
		freqs = np.logspace(np.log10(start_freq), np.log10(end_freq), duration*framerate-1)
		dts = np.diff(self.x_coords)
		dphis = 2*np.pi*freqs*dts+offset
		phases = np.cumsum(dphis)
		phases = np.insert(phases, 0, 0)
		self.y_coords = self.amp*np.sin(phases)

class UncorrelatedUniformNoise(Wave):
	"""docstring for UncorrelatedUniformNoise"""
	def __init__(self, framerate = 11025, amp = 1):
		super(UncorrelatedUniformNoise, self).__init__(framerate)
		self.amp = amp

		self.x_coords = np.linspace(0, 1, framerate)
		self.y_coords = np.random.uniform(-amp, amp, framerate)

class BrownNoise(Wave):
	"""docstring for BrownNoise"""
	def __init__(self, framerate = 11025, amp = 1):
		super(BrownNoise, self).__init__(framerate)
		self.amp = amp

		self.x_coords = np.linspace(0, 1, framerate)
		tmp = np.random.uniform(-1, 1, framerate)
		tmp = np.cumsum(tmp)
		self.y_coords = 2*amp*((tmp-tmp.min())/(tmp.max()-tmp.min())-0.5)

class PinkNoise(Wave):
	"""docstring for PinkNoise"""
	def __init__(self, framerate = 11025, amp = 1, beta = 1):
		super(PinkNoise, self).__init__(framerate)
		self.amp = amp

		## generate white noise
		uun = UncorrelatedUniformNoise(framerate, amp)
		uun_ys = uun.y_coords

		## apply pink filter to frequencies
		freqs, amps = uun.get_frequencies(full = True)
		denom = freqs**(beta/2)
		denom[0] = 1
		amps /= denom

		## transform to time domain
		self.y_coords = np.fft.ifft(amps)
		self.x_coords = np.arange(len(self.y_coords))/framerate


if __name__ == '__main__':
	wave1 = SineSignal(freq = 20).segment(duration = 3.2)
	wave2 = SquareSignal(freq = 20).segment(duration = 3.7)
	wave3 = TriangleSignal(freq = 20).segment(duration = 3.4)
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



