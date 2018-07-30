# -*- coding: UTF-8 -*
import os
import numpy
import skimage.color
import PIL.Image


def loaddata(path):
    print("loading pictures...")
    rootpath = path
    directory = os.listdir(rootpath)
    data = []
    tag = []

    for dir in directory:
        #if not dir is None:
        subpath = os.path.join(path, dir)
        print(subpath)
        cnt = 0
        if os.path.isdir(subpath):
            for subdir in os.listdir(subpath):
                cnt += 1
                print(cnt)
                subsubpath = os.path.join(subpath, subdir)
                # load image
                img = loadimage(subsubpath)
                if img is None:
                    continue
                img = numpy.array(img)
                img = (skimage.color.rgb2gray(img) * 255).astype(numpy.uint8)
                data.append(img)
                tag.append(dir)
    return data, tag


def loadimage(path):
    try:
        image = PIL.Image.open(path)
    except:
        return None

    if image.mode != "RGB":
        image = image.convert("RGB")
    return image;
