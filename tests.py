import os
import numpy
import base
import skimage
def test(path):
	os.chdir(path)
	cwd=os.getcwd()
	directory = os.listdir(cwd)
	myfeatures=[]
	mytype=[]
	for dir in directory:
		if dir!="null":
			subpath = os.path.join(path, dir)
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
					mytype.append(dir)
	return myfeatures, mytype
