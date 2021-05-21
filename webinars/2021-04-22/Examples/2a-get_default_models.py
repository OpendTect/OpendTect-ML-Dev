# -*- coding: utf-8 -*-
"""
This is a test script file.
"""

from dgbpy import dgbkeras
from dgbpy import mlio as dgbmlio

pickssetexfilenm = 'P:\\Git\\python-dgb-ml\\exampledata\\malenov_input_8x8x16_subsample_dBG.h5'
pickssetinfos = dgbmlio.getInfo(pickssetexfilenm, quick=True)
pickssetmodel = dgbkeras.getDefaultModel(
    pickssetinfos, type=dgbkeras.mltypes[dgbkeras.letnetidx][0])
pickssetmodel.summary()


wllexfilenm = 'P:\\Git\\python-dgb-ml\\exampledata\\Log_-_Lithology_supervised_prediction.h5'
wllinfos = dgbmlio.getInfo(wllexfilenm, quick=True)
wllmodel = dgbkeras.getDefaultModel(
    wllinfos, type=dgbkeras.mltypes[dgbkeras.letnetidx][0])
wllmodel.summary()

imgexfilenm = 'P:\\Git\\python-dgb-ml\\exampledata\\Layers_input_2D.h5'
imginfos = dgbmlio.getInfo(imgexfilenm, quick=True)
imgmodel = dgbkeras.getDefaultModel(
    imginfos, type=dgbkeras.mltypes[dgbkeras.unetidx][0])
imgmodel.summary()


lenetmodel = dgbkeras.getDefaultLeNet(False, (5, 9, 17, 3), 1)
lenetmodel.summary()


unetmodel = dgbkeras.getDefaultUnet(False, (128, 128, 128, 1), 1)
unetmodel.summary()
