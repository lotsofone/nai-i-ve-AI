# -*- coding: UTF-8 -*
from sklearn.externals import joblib
from sklearn import svm


def trainmodel(data, typ):
    print("svm fitting")
    clf = svm.SVC(kernel='linear')
    # print(data)
    # print(typ)
    clf.fit(data, typ)
    print("svm fit done")
    joblib.dump(clf, "SVM.pkl")
