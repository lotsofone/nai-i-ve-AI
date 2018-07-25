import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import skimage.color
import cv2
import math
import featurePCA
from skimage.feature import local_binary_pattern
from skimage.feature import greycomatrix, greycoprops

#计算glcm矩阵
#传入矩阵
def glcm (im):
	glcm = greycomatrix(im, [5], [0], 256, symmetric=True, normed=True)
	cont = greycoprops(glcm, 'contrast')
	diss = greycoprops(glcm, 'dissimilarity')
	homo = greycoprops(glcm, 'homogeneity')
	eng = greycoprops(glcm, 'energy')
	corr = greycoprops(glcm, 'correlation')
	ASM = greycoprops(glcm, 'ASM')
	return (cont, diss, homo, eng, corr, ASM)





def g_l_feature(img):
	img=cv2.resize(img, (64,64))
	mp=[]

	#glcm
	cont, diss, homo, eng, corr, ASM=glcm (img)
	mp.append(cont)
	mp.append(diss)
	mp.append(homo)
	mp.append(eng)
	mp.append(corr)
	mp.append(ASM)
	#

	#lbp
	radius = 3
	n_points = 8 * radius
	lbp = local_binary_pattern(img, n_points, radius)
	mp.append(lbp)
	#

	mp=featurePCA.flattenToArray(mp)
	#print(mp)
	return mp



