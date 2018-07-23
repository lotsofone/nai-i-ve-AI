import PIL.Image
import base
import img_factors
import numpy
import skimage
import matplotlib.pyplot
import glcm_lbp
import csv
import os
import pandas
import pickle
def writetocsv(path, data):
	#newline=""是为了防止输出空行
  with open(path,"a+", newline="") as csvfile: 
    writer = csv.writer(csvfile,dialect = ("excel"))
    for row in data:
      writer.writerow(row)
  csvfile.close()


def pic(path):
	os.chdir(path)
	cwd=os.getcwd()
	directory = os.listdir(cwd)
	
	for dir in directory:
		if dir>"puppy":
			subpath = os.path.join(path, dir)
			myfeatures=[]
			print(subpath)
			cnt=0
			if os.path.isdir(subpath):
				for subdir in os.listdir(subpath):
					cnt+=1
					print(cnt)
					subsubpath=os.path.join(subpath, subdir)
					#load image
					img = numpy.array(base.loadimage(subsubpath))
					#print(subsubpath)
					img = (skimage.color.rgb2gray(img)*255).astype(numpy.uint8)
			        #features
					myfeatures.append(base.feature(img))
				load="C:\\Users\\asus-pc\\Documents\\GitHub\\nai-i-ve-AI\\"+dir+".pkl"

				#writetocsv(load, myfeatures)
				wf = open(load, 'wb')
				pickle.dump(myfeatures, file=wf)
				#data=pandas.DataFrame(myfeatures)
				#data.to_csv("C:\\Users\\asus-pc\\Documents\\GitHub\\nai-i-ve-AI\\"+dir+".csv")


numpy.set_printoptions(threshold = 1e6)
pic("D:\\ds2018")

"""
img = numpy.array(base.loadimage("C:\\Users\\asus-pc\\Desktop\\10.jpg"))
img = (skimage.color.rgb2gray(img)*255).astype(numpy.uint8)
gg = img_factors.histogramGray(img)
print(img_factors.histStats(gg))


matplotlib.pyplot.plot(numpy.arange(256),gg , color="#FF0000")
matplotlib.pyplot.show()
print(base.predict("dwad"))
"""