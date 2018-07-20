import random
import PIL.Image
import numpy
import img_factors
import glcm_lbp

types = ['bear','bird','car','cow','elk','fox','giraffe','horse','koala','lion','monkey','plane','puppy','sheep','statue','tiger','tower','train','whale','zebra']

def loadimage(path):
    image = PIL.Image.open(path)
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image;


def feature(img):
	gg=img_factors.histogramGray(img)
	fea1=img_factors.histStats(gg)
	fea2=glcm_lbp.g_l_feature(img)
	fea=list(fea1.values())
	fea.append(list(fea2.values()))
	return fea
	#print(fea)


def predict(path):
    return types[random.randint(0, 20)]

def train(path, type):
    return