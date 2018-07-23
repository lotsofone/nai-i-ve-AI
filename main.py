import PIL.Image
import base
import img_factors
import numpy
import skimage
import matplotlib.pyplot
import glcm_lbp
import featurePCA
import csv
import os
import pandas

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
				print(subsubpath)
				img = (skimage.color.rgb2gray(img)*255).astype(numpy.uint8)
		        #features
				myfeatures.append(base.feature(img))
			data=pandas.DataFrame(myfeatures)
			data.to_csv("C:\\Users\\asus-pc\\Documents\\GitHub\\nai-i-ve-AI\\"+dir+".csv")


#numpy.set_printoptions(threshold = 1e6)
#pic("D:\\ds2018")
featurePCA.loaddataset()