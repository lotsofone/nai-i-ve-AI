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
import pickle
from sklearn.externals import joblib
import SVM
import neural_network
import tests

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
		if dir>"bird":
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
					#print(base.feature(img))
				load="C:\\Users\\asus-pc\\Documents\\GitHub\\nai-i-ve-AI\\p1 features\\"+dir+".pkl"

				#writetocsv(load, myfeatures)
				wf = open(load, 'wb')
				pickle.dump(myfeatures, file=wf)
				#data=pandas.DataFrame(myfeatures)
				#data.to_csv("C:\\Users\\asus-pc\\Documents\\GitHub\\nai-i-ve-AI\\"+dir+".csv")


#numpy.set_printoptions(threshold = 1e6)

#step1
#提取特征向量
pic("D:\\ds2018")

#step2
#降维，并保存pca
#featurePCA.loaddataset()

#step3
#训练svm
"""
wf = open("dataarrays.pkl", 'rb')
data=pickle.load(wf)
wf.close()
wf = open("set.pkl", 'rb')
typ=pickle.load(wf)
wf.close()
print("svm fitting")
clf=SVM.SVM_fit(data, typ)
wf=open("SVM.pkl", 'wb')
pickle.dump(clf, file=wf)
wf.close()
"""

#step4提取测试集的特征向量
"""
fe, ty=tests.test("C:\\Users\\asus-pc\\Desktop\\test")
print(len(fe))
pca=joblib.load("C:\\Users\\asus-pc\\Documents\\GitHub\\nai-i-ve-AI\\pca.pkl")
pca.transform(fe)
wf = open("C:\\Users\\asus-pc\\Documents\\GitHub\\nai-i-ve-AI\\test_features.pkl", 'wb')
pickle.dump(fe, file=wf)
wf.close()
wf = open("C:\\Users\\asus-pc\\Documents\\GitHub\\nai-i-ve-AI\\test_features_answer.pkl", 'wb')
pickle.dump(ty, file=wf)
wf.close()
"""

#step5
#预测测试集
"""
types = ['bear','bicycle','bird','car','cow','elk','fox','giraffe','horse','koala','lion','monkey','plane','puppy','sheep','statue','tiger','tower','train','whale','zebra']
svm=joblib.load("C:\\Users\\asus-pc\\Documents\\GitHub\\nai-i-ve-AI\\SVM.pkl")
wf = open("C:\\Users\\asus-pc\\Documents\\GitHub\\nai-i-ve-AI\\test_features.pkl", 'rb')
fe=pickle.load(wf)
wf.close()
wf = open("C:\\Users\\asus-pc\\Documents\\GitHub\\nai-i-ve-AI\\test_features_answer.pkl", 'rb')
ty=pickle.load(wf)
wf.close()


ret=svm.predict(fe)
#print(ret)
#print(confusion_matrix(ty, ret))  

dec=svm.decision_function(fe)
#print(dec)

c,k=dec.shape
print(c)
print(k)
num=0
for i in range(0, c):
	flag=0
	for j in range(0,5):
		maxx=numpy.max(dec[i])
		for kk in range(0, k):
			if maxx==dec[i][kk]:
				dec[i][kk]=-999
				if types[kk]==ty[i]:
					flag=1
	if flag==1:
		num+=1

print(num/c)
"""


