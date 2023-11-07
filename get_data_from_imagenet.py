import os
root = '/data/xhn/ood/CLIPN/src/ImageNet/train/'

class_img = []

for cls in class_img:
    root_ =  root + cls
    imgs = os.listdir(root_)
    pass