# -*- coding: utf-8 -*-
# @Author: ZMJ
# @Date:   2020-04-07 15:17:20
# @Last Modified by:   ZMJ
# @Last Modified time: 2020-04-09 12:09:02
import sys
sys.path.append("../")
from basic_waves import UncorrelatedUniformNoise, BrownNoise, PinkNoise
from matplotlib import pyplot as plt
import numpy as np

uun = UncorrelatedUniformNoise()
brown = BrownNoise()
pink = PinkNoise()

plt.subplot(2, 4, 1)
plt.plot(uun.x_coords, uun.y_coords)
plt.xlabel("Time (s)")
plt.ylabel("Intensity")
plt.subplot(2, 4, 2)
freqs, amps = uun.get_frequencies()
plt.plot(freqs, amps)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.subplot(2, 4, 3)
freqs, powers = uun.get_power()
plt.plot(freqs, powers)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power")
plt.subplot(2, 4, 4)
freqs, cumpowers = uun.get_cumsum_power()
plt.plot(freqs, cumpowers)
plt.xlabel("Frequency (Hz)")
plt.ylabel("CumSumPower")
plt.subplot(2, 4, 5)
plt.plot(brown.x_coords, brown.y_coords)
plt.xlabel("Time (s)")
plt.ylabel("Intensity")
plt.subplot(2, 4, 6)
freqs, amps = brown.get_frequencies()
plt.plot(freqs, amps)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.subplot(2, 4, 7)
freqs, powers = brown.get_power()
plt.plot(freqs, powers)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power")
plt.subplot(2, 4, 8)
plt.plot(np.log(freqs), np.log(powers))
freqs, powers = pink.get_power()
plt.plot(np.log(freqs), np.log(powers))
freqs, powers = uun.get_power()
plt.plot(np.log(freqs), np.log(powers))
plt.legend(("Brown", "Pink", "White"))
plt.xlabel("Ln Frequency (Hz)")
plt.ylabel("Ln Power")
plt.show()

