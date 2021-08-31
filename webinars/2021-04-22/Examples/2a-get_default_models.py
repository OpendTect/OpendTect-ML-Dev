# -*- coding: utf-8 -*-
"""
This is a test script file.
"""

import os

from dgbpy import dgbkeras
from dgbpy import mlio as dgbmlio
from dgbpy import keystr as dgbkeys

expath = os.path.join(os.path.dirname(os.path.abspath(__file__)),'data')

def getExampleInfos( infodict ):
    learntype = infodict[dgbkeys.learntypedictstr]
    classification = infodict[dgbkeys.classdictstr]
    inpshape = infodict[dgbkeys.inpshapedictstr]
    if isinstance(inpshape,int):
        nrdims = inpshape
    else:
        nrdims = len(inpshape)
    return (learntype,classification,nrdims)

pickssetexfilenm = os.path.join(expath,'malenov_input_8x8x16_subsample_dBG.h5')
pickssetinfos = dgbmlio.getInfo(pickssetexfilenm, quick=True)
(learntype,classification,nrdims) = getExampleInfos( pickssetinfos )
modeltype = dgbkeras.getModelsByType( learntype, classification, nrdims )
pickssetmodel = dgbkeras.getDefaultModel( pickssetinfos, type=modeltype[0] )
pickssetmodel.summary()

wllexfilenm = os.path.join(expath,'Log_-_Lithology_supervised_prediction.h5')
wllinfos = dgbmlio.getInfo(wllexfilenm, quick=True)
(learntype,classification,nrdims) = getExampleInfos( wllinfos )
modeltype = dgbkeras.getModelsByType( learntype, classification, nrdims )
wllmodel = dgbkeras.getDefaultModel( wllinfos, type=modeltype[0] )
wllmodel.summary()

imgexfilenm = os.path.join(expath,'Layers_input_2D.h5')
imginfos = dgbmlio.getInfo(imgexfilenm, quick=True)
(learntype,classification,nrdims) = getExampleInfos( imginfos )
modeltype = dgbkeras.getModelsByType( learntype, classification, nrdims )
imgmodel = dgbkeras.getDefaultModel( imginfos, type=modeltype[0] )
imgmodel.summary()

import dgbpy.keras_classes as kc
if len(kc.UserModel.mlmodels) < 1:
    kc.UserModel.mlmodels = kc.UserModel.findModels()

learntype = dgbkeys.seisclasstypestr
nrattribs = 3
inp_shape = (5, 9, 17)
model_shape = (*inp_shape,nrattribs)
classification = True
nrout = 9 # Number of labels

data_format = 'channels_last'
modeltype = dgbkeras.getModelsByType( learntype, classification, len(inp_shape) )
learnrate = 1e-5
lenetmodel = kc.UserModel.findName(modeltype[0]).model( model_shape, nrout, \
                                            learnrate, data_format=data_format ) 
lenetmodel.summary()

learntype = dgbkeys.seisimgtoimgtypestr
nrattribs = 1
inp_shape = (128, 128, 128)
model_shape = (*inp_shape,nrattribs)
classification = False
nrout = 1

modeltype = dgbkeras.getModelsByType( learntype, classification, len(inp_shape) )
unetmodel = kc.UserModel.findName(modeltype[0]).model( model_shape, nrout, \
                                            learnrate, data_format=data_format ) 
unetmodel.summary()
