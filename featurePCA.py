import sklearn.decomposition.pca
import os
import csv
import numpy
import sklearn.decomposition.pca
import pickle


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
    for filename in os.listdir(sourceRootDir):
        with open(os.path.join(sourceRootDir, filename), 'rb') as sourceFile:
            loadeddata = pickle.load(sourceFile)
            for i in range(len(loadeddata)):
                loadeddata[i] = flattenToArray(loadeddata[i])
                Yset.append(filename)
            dataarrays.extend(loadeddata)

    print(len(dataarrays[0]))
    pca = sklearn.decomposition.pca.PCA()
    print("fitting")
    pca.fit(dataarrays, Yset)
    print("fitdone")
    print(pca.explained_variance_ratio_)
    print(pca.singular_values_)
    print(len(pca.explained_variance_ratio_))
    print(len(pca.singular_values_))
