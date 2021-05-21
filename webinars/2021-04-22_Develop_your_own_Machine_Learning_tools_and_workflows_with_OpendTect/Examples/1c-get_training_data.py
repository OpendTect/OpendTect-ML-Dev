# -*- coding: utf-8 -*-
"""
This is a test script file.
"""

from dgbpy import mlapply as dgbmlapply

pickssetexfilenm = 'P:\\Git\\python-dgb-ml\\exampledata\\malenov_input_8x8x16_subsample_dBG.h5'
pickssetdp = dgbmlapply.getScaledTrainingData(pickssetexfilenm, split=0.2)


wllexfilenm = 'P:\\Git\\python-dgb-ml\\exampledata\\Log_-_Lithology_supervised_prediction.h5'
wlldp = dgbmlapply.getScaledTrainingData(wllexfilenm, split=0.2)


imgexfilenm = 'P:\\Git\\python-dgb-ml\\exampledata\\Layers_input_2D.h5'
imgdp = dgbmlapply.getScaledTrainingData(imgexfilenm, split=0.2)
