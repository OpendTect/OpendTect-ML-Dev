# -*- coding: utf-8 -*-
"""
This is a test script file.
"""

from dgbpy import mlio as dgbmlio

pickssetexfilenm = 'P:\\Git\\python-dgb-ml\\exampledata\\malenov_input_8x8x16_subsample_dBG.h5'
pickssetdp = dgbmlio.getTrainingData(pickssetexfilenm)


wllexfilenm = 'P:\\Git\\python-dgb-ml\\exampledata\\Log_-_Lithology_supervised_prediction.h5'
wlldp = dgbmlio.getTrainingData(wllexfilenm)


imgexfilenm = 'P:\\Git\\python-dgb-ml\\exampledata\\Layers_input_2D.h5'
imgdp = dgbmlio.getTrainingData(imgexfilenm)
