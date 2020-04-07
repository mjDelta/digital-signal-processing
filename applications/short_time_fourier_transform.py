# -*- coding: utf-8 -*-
# @Author: ZMJ
# @Date:   2020-04-05 20:26:31
# @Last Modified by:   ZMJ
# @Last Modified time: 2020-04-07 15:05:53

import sys
sys.path.append('../')
from basic_waves import Chirp, Wave
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sn
def normalization(arr):
	_max = arr.max()
	_min = arr.min()
	norm = (arr-_min)/(_max-_min)
	return norm

time_length = 256
wave = Chirp()
time_parts = np.arange(0, wave.framerate, time_length)

last = time_parts[0]
for time_part in time_parts[1:]:
	start = last
	end = time_part

	xs = wave.x_coords[start:end]
	ys = wave.y_coords[start:end]
	tmp_wave = Wave(wave.framerate)
	tmp_wave.set_values(xs, ys)
	freqs, amps = tmp_wave.get_frequencies()

	last = time_part

	time = 1/wave.framerate*time_part
	idx = np.argmax(amps)
	print("Timepoint {}'\tmax frequency is {}".format(round(time, 4), freqs[idx]))

