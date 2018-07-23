import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import skimage.color
import cv2
import math

#计算glcm矩阵
#传入矩阵，角度，d
def glcm (im, theta, d, gray_level=16):
	max_level=im.max()+1
	im=im.astype(np.int16)

	if(max_level>16):
		im = (im * gray_level) / max_level
	im=im.astype(np.uint8)
	#print(im)
	c, k= im.shape

	lis=np.zeros((gray_level, gray_level), dtype=np.int16)
	h=int(math.ceil(math.sin(theta)*d))
	w=int(math.ceil(math.cos(theta)*d))

	for i in range(0,c-1):
		for j in range(0, k-1):
			if i+w>0 and i+w<c and j-h>0 and j-h<k:
				ind1=im[i,j]
				ind2=im[i+w,j-h]
				lis[ind1,ind2]+=1
	return lis

#灰度共生矩阵的能量
#传入矩阵
def energy(L):
	#print(L)
	L=L.astype(np.int32)
	tmp=L*L
	energy=np.sum(tmp)
	#print(tmp)
	#print(energy)
	return energy

#灰度共生矩阵的熵
#传入矩阵
def entrop(L):
	#print(L)
	step1=np.log2(L)
	tag=np.isinf(step1)
	step1[tag]=0 
	step2=step1*L
	entrop=-np.sum(step2)
	return entrop
	#print(step1)
	#print(step2)


#原始LBP矩阵
#传入矩阵
def LBP(img):
	c, k=img.shape
	lbp=np.zeros(img.shape)
	mul_matrix=[[1,2,4],[128,0,8],[64,32,16]]
	for i in range(1,c-1):
		for j in range(1, k-1):
			me=img[i, j]
			ret=0
			if img[i-1,j-1]>me:
			    ret=ret|mul_matrix[0][0]*img[i-1, j-1]
			if img[i-1, j]>me: 
				ret=ret|mul_matrix[0][1]*img[i-1, j]
			if img[i-1, j+1]>me: 
				ret=ret|mul_matrix[0][2]*img[i-1, j+1]
			if img[i, j-1]>me:
				ret=ret|mul_matrix[1][0]*img[i,j-1]
			if img[i, j+1]>me:
				ret=ret|mul_matrix[1][2]*img[i, j+1]
			if img[i+1, j-1]>me:
				ret=ret|mul_matrix[2][0]*img[i+1, j-1]
			if img[i+1, j]>me:
				ret=ret|mul_matrix[2][1]*img[i+1,j]
			if img[i+1, j+1]>me:
				ret=ret|mul_matrix[2][2]*img[i+1, j+1]
			#print(lbp[i])
			lbp[i][j]=ret
		#print(lbp[i])
	#print(lbp)
	return lbp



def g_l_feature(img):
	img=cv2.resize(img, (64,64))
	mp={}
	tmp=glcm (img, 0, 1)
	eng=energy(tmp)
	entro=entrop(tmp)
	lbp=LBP(img)
	mp['energy']=eng
	mp['entrop']=entro
	mp['lbp']=lbp
	return mp

#im=np.array(Image.open("C:\\Users\\asus-pc\\Desktop\\10.jpg").convert('L'))
#im=cv2.resize(im,(64,64))
#g_l_feature(im)

