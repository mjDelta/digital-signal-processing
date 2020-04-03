# -*- coding: utf-8 -*-
# @Author: ZMJ
# @Date:   2020-03-28 18:26:34
# @Last Modified by:   ZMJ
# @Last Modified time: 2020-04-03 09:30:15
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

	# def get_frequencies(self):
	# 	fft_size = self.framerate / (1 / self.period)
	# 	fft_size = int(np.ceil(fft_size)) 
	# 	amps = np.fft.rfft(self.y_coords[:fft_size]) / fft_size
	# 	dbs = 20*np.log10(np.clip(np.abs(amps),1e-20,1e100))
	# 	amps = np.abs(amps)
	# 	freqs = np.linspace(0, self.framerate//2, fft_size//2+1)
	# 	return freqs, amps
	def get_frequencies(self):
		"""[summary]
		
		[description]
		
		Arguments:
			fs {[type]} -- [the sampling frequency for fft]
		"""
		freqs = range(0, self.framerate)
		amps = np.abs(np.fft.fft(self.y_coords)*2/self.framerate)
		##normalization
		amps /= self.framerate
		##half
		half = self.framerate//2
		freqs = freqs[:half]
		amps = amps[:half]
		return freqs, amps

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
		print(min_tick, start_pnt, end_pnt)
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

		self.x_coords = np.linspace(0, duration, framerate)
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

		self.x_coords = np.linspace(0, duration, framerate)
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

		self.x_coords = np.linspace(0, duration, framerate)
		cycles = self.freq * self.x_coords + self.offset/(2*np.pi)
		fracs, _ = np.modf(cycles)
		self.y_coords =  np.sign(fracs - 0.5)*self.amp
		
if __name__ == '__main__':
	sine_wave1 = SquareWave(freq = 200, offset = np.pi, amp = 10, framerate = 10000)
	sine_wave2 = SquareWave(freq = 400, offset = np.pi, amp = 30, framerate = 10000)
	sine_wave = sine_wave1 + sine_wave2

	seg1 = sine_wave1.segment(start = 0.1)
	seg2 = sine_wave2.segment(start = 0.1)
	plt.subplot(3,3,1)
	plt.plot(sine_wave1.x_coords, sine_wave1.y_coords)
	plt.subplot(3,3,2)
	plt.plot(sine_wave2.x_coords, sine_wave2.y_coords)
	plt.subplot(3,3,3)
	plt.plot(sine_wave.x_coords, sine_wave.y_coords)
	plt.subplot(3,3,4)
	plt.plot(seg1.x_coords, seg1.y_coords)
	plt.subplot(3,3,5)
	plt.plot(seg2.x_coords, seg2.y_coords)
	plt.subplot(3,3,6)
	plt.plot(sine_wave.x_coords, sine_wave.y_coords)
	plt.xlim(0.1, 0.2)
	plt.subplot(3,3,7)
	freqs, amps = sine_wave1.get_frequencies()
	plt.plot(freqs, amps)
	plt.subplot(3,3,8)
	freqs, amps = sine_wave2.get_frequencies()
	plt.plot(freqs, amps)
	plt.subplot(3,3,9)
	freqs, amps = sine_wave.get_frequencies()
	plt.plot(freqs, amps)
	plt.show()






