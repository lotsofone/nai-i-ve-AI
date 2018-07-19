#某画直方图的
#不要引用
import skimage.filters
import skimage.color
import PIL.Image
import matplotlib.pyplot
import numpy
import cv2
import math
import scipy.signal


def histogramGray(im):
    gcount = numpy.zeros(256)
    def gcc(v):
        gcount[v]+=1
        return 0

    list(map(gcc, im.flatten()))
    gcount/=len(im.flatten())
    return gcount

def histStats(gg):
    props = {}
    avg=0
    for i in range(256):
        avg+=i*gg[i]
    props['avg']=avg
    var=0
    for i in range(256):
        var+=(i-avg)*(i-avg)*gg[i]
    props['var']=var
    max=255
    while max>=0:
        if(gg[max]>0):
            break
        max-=1

    min=0
    while min<256:
        if(gg[min]>0):
            break
        min+=1
    props['max']=max
    props['min']=min
    skewness=0
    for i in range(256):
        skewness+=(i-avg)*(i-avg)*(i-avg)*gg[i]
    skewness/=var
    props['skewness']=skewness
    kurtosis=0
    for i in range(256):
        kurtosis+=(i-avg)*(i-avg)*(i-avg)*(i-avg)*gg[i]
    kurtosis/=var
    kurtosis/=var
    kurtosis-=3
    props['kurtosis']=kurtosis
    energy=0
    for i in range(256):
        energy+=gg[i]*gg[i]
    props['energy']=energy
    entropy = 0
    for i in range(256):
        if(gg[i]>0):
            entropy+=gg[i]+math.log2(gg[i])
    props['entropy']=entropy

    return props
"""
image = loadimage("download\\829.jpg")
im=numpy.array(image)
#im=im.astype(numpy.float32)/255
r = im[:,:,0]//64
g = im[:,:,1]//64
b = im[:,:,2]//64
gray=(skimage.color.rgb2gray(im)*255).astype(numpy.uint8)

rcount = numpy.zeros(4)
gcount = numpy.zeros(4)
bcount = numpy.zeros(4)
graycount = numpy.zeros(256)
def rcc(v):
    rcount[v]+=1
    return 0
list(map(rcc, r.flatten()))
def gcc(v):
    gcount[v]+=1
    return 0
list(map(gcc, g.flatten()))
def bcc(v):
    bcount[v]+=1
    return 0
list(map(bcc, b.flatten()))
def graycc(v):
    graycount[v]+=1
    return 0
list(map(graycc, gray.flatten()))

matplotlib.pyplot.subplot(2,1,1)
matplotlib.pyplot.plot(numpy.arange(4), rcount, color="#FF0000")
#matplotlib.pyplot.subplot(2,2,1)
matplotlib.pyplot.plot(numpy.arange(4), gcount, color="#00FF00")
#matplotlib.pyplot.subplot(2,2,1)
matplotlib.pyplot.plot(numpy.arange(4), bcount, color="#0000FF")
matplotlib.pyplot.subplot(2,1,2)
matplotlib.pyplot.hist(gray.flatten(), 255, color="#888888")
matplotlib.pyplot.show()
"""
