import PIL.Image
import base
import img_factors
import numpy
import skimage
import matplotlib.pyplot



img = numpy.array(base.loadimage("D:\\EUPS\\AI\\download\\829.jpg"))
img = (skimage.color.rgb2gray(img)*255).astype(numpy.uint8)
gg = img_factors.histogramGray(img)
print(img_factors.histStats(gg))
matplotlib.pyplot.plot(numpy.arange(256),gg , color="#FF0000")
matplotlib.pyplot.show()
#print(base.predict("dwad"))