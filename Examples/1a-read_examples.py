# -*- coding: utf-8 -*-
"""
This is a test script file.
"""


from dgbpy import mlio as dgbmlio

pickssetexfilenm = 'P:\\Git\\python-dgb-ml\\exampledata\\malenov_input_8x8x16_subsample_dBG.h5'
pickssetsinfos = dgbmlio.getInfo(pickssetexfilenm, quick=True)

wllexfilenm = 'P:\\Git\\python-dgb-ml\\exampledata\\Log_-_Lithology_supervised_prediction.h5'
wllinfos = dgbmlio.getInfo(wllexfilenm, quick=True)

imgexfilenm = 'P:\\Git\\python-dgb-ml\\exampledata\\Layers_input_2D.h5'
imginfos = dgbmlio.getInfo(imgexfilenm, quick=True)
