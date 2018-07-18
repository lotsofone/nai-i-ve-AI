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

def loadimage(path):
    image = PIL.Image.open(path)
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image;

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