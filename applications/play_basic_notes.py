# -*- coding: utf-8 -*-
# @Author: ZMJ
# @Date:   2020-03-29 18:47:49
# @Last Modified by:   ZMJ
# @Last Modified time: 2020-03-30 22:48:57
import sys
sys.path.append('../')
from basic_waves import SineWave, Wave
import time
import numpy as np

wave1 = SineWave(freq = 0.523, duration = 0.5)
wave2 = SineWave(freq = 0.587, duration = 0.5)
wave3 = SineWave(freq = 659, duration = 0.5)
wave4 = SineWave(freq = 698, duration = 0.5)
wave5 = SineWave(freq = 783, duration = 0.5)
wave6 = SineWave(freq = 880, duration = 0.5)
wave7 = SineWave(freq = 987, duration = 0.5)

notes = [wave1, wave2, wave3, wave4, wave5, wave6, wave7]
musics = "333 333 35123 |444 4433 3332232 5| 333 333 35123 |444 4433 3355421 |53215 5553216| 64327 55423 53215| 5553216 6432555 565421 5 |333 333 35123 |444 4433 3332232 5 |333 333 35123 |444 4433 3355671"

ys=[]
for m in musics:
	if m == " ":
		ys.extend(list(np.zeros((note.framerate//3))))
	elif m == "|":
		ys.extend(list(np.zeros((note.framerate//2))))

	else:
		m = int(m)
		note = notes[m-1]
		ys.extend(list(note.y_coords))

music = Wave(1, note.framerate)
music.set_values(None, np.array(ys))
music.play_wave()
