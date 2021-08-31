# -*- coding: utf-8 -*-
"""
This is a test script file.
"""

import os

from dgbpy import mlapply as dgbmlapply

expath = os.path.join(os.path.dirname(os.path.abspath(__file__)),'data')

pickssetexfilenm = os.path.join(expath,'malenov_input_8x8x16_subsample_dBG.h5')
pickssetdp = dgbmlapply.getScaledTrainingData(pickssetexfilenm, split=0.2)


wllexfilenm = os.path.join(expath,'Log_-_Lithology_supervised_prediction.h5')
wlldp = dgbmlapply.getScaledTrainingData(wllexfilenm, split=0.2)


imgexfilenm = os.path.join(expath,'Layers_input_2D.h5')
imgdp = dgbmlapply.getScaledTrainingData(imgexfilenm, split=0.2)
