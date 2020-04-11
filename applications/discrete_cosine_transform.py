# -*- coding: utf-8 -*-
# @Author: ZMJ
# @Date:   2020-04-11 17:09:31
# @Last Modified by:   ZMJ
# @Last Modified time: 2020-04-11 17:49:08

import sys
sys.path.append("../")
from basic_waves import Wave, TriangleSignal
import numpy as np
import matplotlib.pyplot as plt

amps = np.array([2, 0.6, 0.3, 0.2])
freqs = np.array([100, 200, 250, 300])
framerate = 11025

wave = Wave(framerate)
synthesize_simple = wave.discrete_cosine_synthesize_sumption(amps, freqs)
synthesize_linalg = wave.discrete_cosine_synthesize_linalg(amps, freqs)
print("MAE of synthesizing cosine waves by sumption and linear algebra: {}".format(
	np.sum(np.abs(synthesize_linalg.y_coords-synthesize_simple.y_coords)))
)

triangle = TriangleSignal(freq = 400, framerate = 10000)
freqs, amps = triangle.discrete_consine_tranform_iv()
plt.subplot(2, 1, 1)
# plt.plot(synthesize_simple.x_coords, synthesize_simple.y_coords, label = "Simple sumption")
plt.plot(triangle.x_coords, triangle.y_coords, label = "Linear algebra")
plt.subplot(2, 1, 2)
plt.plot(freqs, amps)
# plt.xlim(0,500)
# plt.legend()
plt.show()


