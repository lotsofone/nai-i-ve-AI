import random
import PIL.Image

types = ['bear','bird','car','cow','elk','fox','giraffe','horse','koala','lion','monkey','plane','puppy','sheep','statue','tiger','tower','train','whale','zebra']


def loadimage(path):
    image = PIL.Image.open(path)
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image;



def predict(path):
    return types[random.randint(0, 20)]

def train(path, type):
    return