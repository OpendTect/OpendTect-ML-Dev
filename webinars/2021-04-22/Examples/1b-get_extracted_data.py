# -*- coding: utf-8 -*-
"""
This is a test script file.
"""

import os

from dgbpy import mlio as dgbmlio

expath = os.path.join(os.path.dirname(os.path.abspath(__file__)),'data')

pickssetexfilenm = os.path.join(expath,'malenov_input_8x8x16_subsample_dBG.h5')
pickssetdp = dgbmlio.getTrainingData(pickssetexfilenm)


wllexfilenm = os.path.join(expath,'Log_-_Lithology_supervised_prediction.h5')
wlldp = dgbmlio.getTrainingData(wllexfilenm)


imgexfilenm = os.path.join(expath,'Layers_input_2D.h5')
imgdp = dgbmlio.getTrainingData(imgexfilenm)
