# -*- coding: utf-8 -*-
# @Author: ZMJ
# @Date:   2020-03-28 18:26:34
# @Last Modified by:   ZMJ
# @Last Modified time: 2020-03-30 22:54:05
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read as wave_reader
from scipy.io.wavfile import write as wave_writer
import pyaudio
class Wave(object):
	"""docstring for ClassName"""
	def __init__(self, period, framerate):
		super(Wave).__init__()
		self.period = period
		self.framerate = framerate

	def segment(self, start=0, duration=5):
		"""[summary]
		
		[description]
		
		Keyword Arguments:
			start {number} -- [the required start point, located between 0 and 1] (default: {0})
			duration {number} -- [the number of period] (default: {5})
		"""

		min_tick = 1 / self.framerate
		start_pnt = int(start / min_tick)
		end_pnt = start_pnt + int(duration / min_tick)

		wave = Wave(self.period, end_pnt - start_pnt)	
		wave.set_values(self.x_coords[start_pnt:end_pnt], self.y_coords[start_pnt:end_pnt])
		return wave

	def __add__(self, wave):
		"""[summary]
		
		[description]
		
		Arguments:
			sinewave {[type]} -- [return the sumaption of two waves]
		"""
		new_y_coords = self.y_coords + wave.y_coords
		new_freq = self.get_GCD(1./self.period, wave.period)
		wave = Wave(1./new_freq, self.framerate)	
		wave.set_values(self.x_coords, new_y_coords)
		
		return wave

	def set_values(self, x_coords, y_coords):
		self.x_coords = x_coords
		self.y_coords = y_coords
		self.y_coords=self.y_coords.astype(np.float32)

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
	def __init__(self, freq = 25, amp = 1., offset = 0., framerate = 11025, duration = 1):
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


if __name__ == '__main__':
	# sine_wave1 = SineWave(offset = np.pi/2)
	# sine_wave2 = SineWave(freq = 30, offset = np.pi)
	# sine_wave = sine_wave1 + sine_wave2

	# seg1 = sine_wave1.segment(start = 0.1)
	# seg2 = sine_wave2.segment(start = 0.1)
	# seg = sine_wave.segment(start = 0.1)
	# print(sine_wave.freq, seg.freq, sine_wave1.freq, sine_wave2.freq)
	# plt.subplot(4,1,1)
	# plt.plot(sine_wave.x_coords, sine_wave.y_coords)
	# plt.subplot(4,1,2)
	# plt.plot(seg1.x_coords, seg1.y_coords)
	# plt.subplot(4,1,3)
	# plt.plot(seg2.x_coords,seg2.y_coords)
	# plt.subplot(4,1,4)
	# plt.plot(seg.x_coords,seg.y_coords)
	# plt.show()

	wave1 = SineWave(freq = 523, duration = 5)
	wave2 = SineWave(freq = 587, duration = 5)
	wave3 = SineWave(freq = 659, duration = 5)
	wave4 = SineWave(freq = 698, duration = 5)
	wave5 = SineWave(freq = 783, duration = 5)
	wave6 = SineWave(freq = 880, duration = 5)
	wave7 = SineWave(freq = 987, duration = 5)

	wave1.play_wave()
	wave2.play_wave()
	wave3.play_wave()
	wave4.play_wave()
	wave5.play_wave()
	wave6.play_wave()
	wave7.play_wave()




