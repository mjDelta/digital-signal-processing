# -*- coding: utf-8 -*-
# @Author: ZMJ
# @Date:   2020-04-10 10:22:04
# @Last Modified by:   ZMJ
# @Last Modified time: 2020-04-10 10:31:31
import sys
sys.path.append("../")
from basic_waves import PinkNoise, SineSignal
import matplotlib.pyplot as plt

w0 = PinkNoise(beta = 0.0)
w1 = PinkNoise(beta = 0.3)
w2 = PinkNoise(beta = 1.0)
w3 = PinkNoise(beta = 1.7)
w4 = SineSignal(freq = 50)

ws = [w0, w1, w2, w3, w4]
names = ["Beta = 0.", "Beta = 0.3", "Beta = 1.0", "Beta = 1.7", "Sine (freq =50)"]
for w, n in zip(ws, names):
	lags, corrs = w.compute_autocorr()
	plt.plot(lags, corrs, label = n)
plt.legend()
plt.show()
