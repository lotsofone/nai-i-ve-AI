# -*- coding: UTF-8 -*
import PIL.Image
import skimage
import matplotlib.pyplot
import cv2
import os
import pickle
from sklearn.externals import joblib
from sklearn.cluster import MiniBatchKMeans
import sklearn.decomposition.pca
from sklearn import svm
import numpy as np
from skimage import feature as ft
from skimage.feature import local_binary_pattern
from skimage.feature import greycomatrix, greycoprops
from sklearn.cluster import KMeans
import math


def extractfeature(data, tags):
    print("extract features....")
    get_feature(data, tags)
    train_vocabulary(data)
    all_feature, typ = pca_train()
    return all_feature, typ


def get_feature(data, tags):
    # print(data)
    num = len(data)
    myfeatures_1 = []
    myfeatures_2 = []
    myfeatures_3 = []
    myfeatures_4 = []
    typ = []
    featureSet = np.float32([]).reshape(0, 128)

    for i in range(0, num):
        img = data[i]
        des = calcSiftFeature(img)
        featureSet = np.append(featureSet, des, axis=0)
        myfeatures_1.append(hist(img))
        myfeatures_2.append(glcm_feature(img))
        myfeatures_3.append(lbp_feature(img))
        myfeatures_4.append(hog(img))
        typ.append(tags[i])

    pickle.dump(myfeatures_1, open("train_feature1.pkl", "wb"))
    pickle.dump(myfeatures_2, open("train_feature2.pkl", "wb"))
    pickle.dump(myfeatures_3, open("train_feature3.pkl", "wb"))
    pickle.dump(myfeatures_4, open("train_feature4.pkl", "wb"))
    pickle.dump(featureSet, open("sift-bow.pkl", "wb"))
    pickle.dump(typ, open("typ.pkl", "wb"))


def train_vocabulary(data):
    print("start training vocabulary...")
    featureSet = pickle.load(open("sift-bow.pkl", "rb"))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.1)
    flags = cv2.KMEANS_RANDOM_CENTERS
    mbk = MiniBatchKMeans(n_clusters=50, batch_size=1000)
    mbk.fit(featureSet)
    centers = mbk.cluster_centers_
    # save vocabulary(a tuple of (labels, centers)) to file
    filename = "voc.pkl"
    savefile(centers, filename)
    print("done")

    num = len(data)
    trainData = np.float32([]).reshape(0, 50)
    for i in range(0, num):
        img = data[i]
        features = calcSiftFeature(img)
        featVec = calcFeatVec(features, centers)
        trainData = np.append(trainData, featVec, axis=0)
    pickle.dump(trainData, open("train_feature5.pkl", "wb"))


#############

def pca_train():
    myfeatures_1 = pickle.load(open("train_feature1.pkl", "rb"))
    myfeatures_2 = pickle.load(open("train_feature2.pkl", "rb"))
    myfeatures_3 = pickle.load(open("train_feature3.pkl", "rb"))
    myfeatures_4 = pickle.load(open("train_feature4.pkl", "rb"))
    myfeatures_5 = pickle.load(open("train_feature5.pkl", "rb"))
    typ = pickle.load(open("typ.pkl", "rb"))

    print("pca start")
    pca1 = sklearn.decomposition.pca.PCA(n_components=100)
    myfeatures_1 = pca1.fit_transform(myfeatures_1)
    joblib.dump(pca1, "pca1.pkl")
    print("pca 1 done")

    pca2 = sklearn.decomposition.pca.PCA(n_components=300)
    myfeatures_2 = pca2.fit_transform(myfeatures_2)

    joblib.dump(pca2, "pca2.pkl")
    print("pca 2 done")

    pca3 = sklearn.decomposition.pca.PCA(n_components=300)
    myfeatures_3 = pca3.fit_transform(myfeatures_3)
    joblib.dump(pca3, "pca3.pkl")
    print("pca 3 done")

    pca4 = sklearn.decomposition.pca.PCA(n_components=300)
    myfeatures_4 = pca4.fit_transform(myfeatures_4)
    joblib.dump(pca4, "pca4.pkl")
    print("pca 4 done")

    ###########
    all_feature = np.array(myfeatures_1)
    all_feature = np.hstack((all_feature, myfeatures_2))
    all_feature = np.hstack((all_feature, myfeatures_3))
    all_feature = np.hstack((all_feature, myfeatures_4))
    all_feature = np.hstack((all_feature, myfeatures_5))
    all_feature = meaningful(np.array(all_feature))

    return all_feature, typ


############################################

def hog(img):
    img = cv2.resize(img, (128, 128))
    features = ft.hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3), visualize=False,
                      transform_sqrt=False, feature_vector=True)
    # print(len(features))
    return features


def calcSiftFeature(img):
    sift = cv2.xfeatures2d.SIFT_create(200)
    kp, des = sift.detectAndCompute(img, None)
    return des


def calcFeatVec(features, centers):
    featVec = np.zeros((1, 50))
    for i in range(0, features.shape[0]):
        fi = features[i]
        diffMat = np.tile(fi, (50, 1)) - centers
        sqSum = (diffMat ** 2).sum(axis=1)
        dist = sqSum ** 0.5
        sortedIndices = dist.argsort()
        idx = sortedIndices[0]  # index of the nearest center
        featVec[0][idx] += 1
    return featVec


def hist(img):
    hist = cv2.calcHist(img, [0], None, [256], [0.0, 255.0])
    hist = flattenToArray(hist)
    return hist


def glcm(im):
    glcm = greycomatrix(im, [2, 8, 16], [0, np.pi / 2.0, np.pi * 3 / 4.0], 32, symmetric=True, normed=True)
    cont = greycoprops(glcm, 'contrast')
    diss = greycoprops(glcm, 'dissimilarity')
    homo = greycoprops(glcm, 'homogeneity')
    eng = greycoprops(glcm, 'energy')
    corr = greycoprops(glcm, 'correlation')
    ASM = greycoprops(glcm, 'ASM')
    return (cont, diss, homo, eng, corr, ASM, glcm)


# glcm
def glcm_feature(img):
    img = cv2.resize(img, (128, 128))
    img = img // 8
    cont, diss, homo, eng, corr, ASM, gl = glcm(img)
    mp = []
    mp.append(cont)
    mp.append(diss)
    mp.append(homo)
    mp.append(eng)
    mp.append(corr)
    mp.append(ASM)
    mp.append(gl)
    mp = flattenToArray(mp)
    # print("siz=")
    # print(len(mp))
    return mp;


# lbp
def lbp_feature(img):
    img = cv2.resize(img, (128, 128))
    mp = []
    # lbp 3 8
    radius = 6
    n_points = 8 * radius
    lbp = local_binary_pattern(img, n_points, radius)
    mp.append(lbp)
    mp = flattenToArray(mp)
    return mp


def flattenToArray(l):
    newlist = []
    for object in l:
        if type(object) == np.ndarray:
            object = object.flatten()
            newlist.extend(object.tolist())
        elif type(object) == list:
            newlist.extend(flattenToArray(object))
        else:
            newlist.append(object)
    return newlist


def meaningful(dataset):
    avg = np.zeros(dataset.shape[1])
    for i in range(len(avg)):
        avg[i] = np.sum(dataset[:, i])
    avg /= float(dataset.shape[0])
    dataset = dataset - [avg]

    miu = np.zeros(dataset.shape[1]).astype(np.float32)
    for i in range(len(miu)):
        miu[i] = np.var(dataset[:, i])
        if miu[i] == 0:
            miu[i] = 1
        else:
            miu[i] = math.sqrt(miu[i])
    dataset = dataset / [miu]

    return dataset


def savefile(data, filename):
    wf = open(filename, 'wb')
    pickle.dump(data, wf)
    wf.close()


def openfile(filename):
    wf = open(filename, 'rb')
    file = pickle.load(wf)
    wf.close()
    return file
