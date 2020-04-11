# -*- coding: utf-8 -*-
# @Author: ZMJ
# @Date:   2020-04-11 17:09:31
# @Last Modified by:   ZMJ
# @Last Modified time: 2020-04-11 17:18:18

import sys
sys.path.append("../")
from basic_waves import Wave
import numpy as np
import matplotlib.pyplot as plt

amps = np.array([2, 0.6, 0.3, 0.2])
freqs = np.array([10, 20, 25, 30])
framerate = 11025

wave = Wave(framerate)
synthesize_simple = wave.discrete_cosine_synthesize_sumption(amps, freqs)
synthesize_linalg = wave.discrete_cosine_synthesize_linalg(amps, freqs)
print("MAE of synthesizing cosine waves by sumption and linear algebra: {}".format(
	np.sum(np.abs(synthesize_linalg.y_coords-synthesize_simple.y_coords)))
)

plt.plot(synthesize_simple.x_coords, synthesize_simple.y_coords, label = "Simple sumption")
plt.plot(synthesize_linalg.x_coords, synthesize_linalg.y_coords, label = "Linear algebra")
plt.legend()
plt.show()


