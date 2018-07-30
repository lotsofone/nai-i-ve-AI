from load import loaddata
from feature import extractfeature
from classify import trainmodel
from predict import testmodel

from load import loadimage
#loadimage("D:\\EUPS\\AI\\ds2018\\bicycle\\0bc137c0-8747-11e8-8dc6-1cb72c9340d7.jpg")
#data, tags=loaddata("E:\\test2")
#features, typ=extractfeature(data, tags)
#trainmodel(features, typ)
res=testmodel("E:\\test2")
print("precide:")
#print(res)