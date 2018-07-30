# -*- coding: UTF-8 -*

from load import *
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


def testmodel(path):
    data, tags = loaddata(path)
    types = ['bear', 'bicycle', 'bird', 'car', 'cow', 'elk', 'fox', 'giraffe', 'horse', 'koala', 'lion', 'monkey',
             'plane', 'puppy', 'sheep', 'statue', 'tiger', 'tower', 'train', 'whale', 'zebra']
    svm = joblib.load("SVM.pkl")
    pca1 = joblib.load("pca1.pkl")
    pca2 = joblib.load("pca2.pkl")
    pca3 = joblib.load("pca3.pkl")
    pca4 = joblib.load("pca4.pkl")
    voc = joblib.load("voc.pkl")
    centers = joblib.load("voc.pkl")
    testfeatures_1 = []
    testfeatures_2 = []
    testfeatures_3 = []
    testfeatures_4 = []
    testfeatures_5 = []
    trainData = np.float32([]).reshape(0, 50)
    ty = []
    ######
    num = len(data)
    # print("!")
    for i in range(0, num):
        img = data[i]
        ty.append(tags[i])
        testfeatures_1.append(hist(img))
        testfeatures_2.append(glcm_feature(img))
        testfeatures_3.append(lbp_feature(img))
        testfeatures_4.append(hog(img))
        features = calcSiftFeature(img)
        featVec = calcFeatVec(features, centers)
        # print(len(featVec))
        trainData = np.append(trainData, featVec, axis=0)

    #####pca
    testfeatures_1 = pca1.transform(testfeatures_1)
    testfeatures_2 = pca2.transform(testfeatures_2)
    testfeatures_3 = pca3.transform(testfeatures_3)
    testfeatures_4 = pca4.transform(testfeatures_4)
    testfeatures_5 = trainData

    all_feature = np.array(testfeatures_1)
    all_feature = np.hstack((all_feature, testfeatures_2))
    all_feature = np.hstack((all_feature, testfeatures_3))
    all_feature = np.hstack((all_feature, testfeatures_4))
    all_feature = np.hstack((all_feature, testfeatures_5))
    all_feature = meaningful(np.array(all_feature))

    ret = svm.predict(all_feature)
    dec = svm.decision_function(all_feature)
    # print(dec)
    c, k = dec.shape
    num = 0
    for i in range(0, c):
        flag = 0
        # print(ty[i])
        for j in range(0, 5):
            maxx = np.max(dec[i])
            for kk in range(0, k):
                if maxx == dec[i][kk]:
                    dec[i][kk] = -999
                    if types[kk] == ty[i]:
                        flag = 1
        if flag == 1:
            num += 1
    # print(num)
    # print(c)
    # print(num/c)
    return (num / float(c))


##############################
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

"""
def loaddata(path):
    print("loading pictures...")
    rootpath = path
    directory = os.listdir(rootpath)
    data = []
    tag = []

    for dir in directory:
        if not dir is None:
            subpath = os.path.join(path, dir)
            print(subpath)
            cnt = 0
            if os.path.isdir(subpath):
                for subdir in os.listdir(subpath):
                    cnt += 1
                    print(cnt)
                    subsubpath = os.path.join(subpath, subdir)
                    # load image
                    img = np.array(loadimage(subsubpath))
                    img = (skimage.color.rgb2gray(img) * 255).astype(np.uint8)
                    data.append(img)
                    tag.append(dir)
    return data, tag


def loadimage(path):
    image = PIL.Image.open(path)
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image;
"""