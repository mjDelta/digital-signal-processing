# -*- coding: utf-8 -*-
# @Author: ZMJ
# @Date:   2020-03-29 18:47:49
# @Last Modified by:   ZMJ
# @Last Modified time: 2020-03-29 18:50:22
import sys
sys.path.append('../')
from basic_waves import SineWave

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