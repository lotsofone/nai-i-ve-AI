import sklearn.decomposition.pca
import os
import csv
import numpy
import sklearn.decomposition.pca
import pickle
from sklearn.externals import joblib


def flattenToArray(l):
    newlist = []
    for object in l:
        if type(object) == numpy.ndarray:
            object = object.flatten()
            newlist.extend(object.tolist())
        elif type(object) == list:
            newlist.extend(flattenToArray(object))
        else:
            newlist.append(object)
    return newlist


def loaddataset():
    sourceRootDir = "p1 features"
    destRootDir = "p2 PCA feature"
    dataarrays = []
    Yset = []
    print("reading")
    for filename in os.listdir(sourceRootDir):
        with open(os.path.join(sourceRootDir, filename), 'rb') as sourceFile:
            loadeddata = pickle.load(sourceFile)
            for i in range(len(loadeddata)):
                loadeddata[i] = flattenToArray(loadeddata[i])
                Yset.append(filename[0:-4])
            dataarrays.extend(loadeddata)
            #print(loadeddata)

    #print(len(dataarrays[0]))
    pca = sklearn.decomposition.pca.PCA(n_components=1000, copy=False)
    #pca = sklearn.decomposition.pca.PCA(copy=True)
    print(len(dataarrays))
    print(len(dataarrays[0]))
    print(len(dataarrays[1]))
    print("fitting")
    pca.fit(dataarrays) 
    print("fitdone")
    print(pca.explained_variance_ratio_)
    print(pca.singular_values_)
    print(len(pca.explained_variance_ratio_))
    print(len(pca.singular_values_))

    #save pca

    joblib.dump(pca,"pca.pkl")

    #save data
    wf=open("dataarrays.pkl", 'wb')
    pickle.dump(dataarrays, file=wf)
    wf.close()

    #save type
    wf=open("set.pkl", 'wb')
    pickle.dump(Yset, file=wf)
    wf.close()