from sklearn import svm

def SVM_fit(data, typ):
	clf = svm.SVC(kernel='linear')
	clf.fit(data, typ)
	return clf

def SVM_predict(unknown, clf):
	typ=clf.predict(unknown)
	print(typ)
