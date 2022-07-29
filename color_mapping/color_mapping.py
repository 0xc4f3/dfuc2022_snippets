import os
import numpy as np
import cv2 as cv
import PIL
import matplotlib.pyplot as plt
    
from os import path
from glob import glob


PATH_IN = './in/'
PATH_OUT = './out/'

mapping = {
    (0, 0, 0): (0, 0, 0),
    (1, 1, 1): (255, 255, 255), # 1 -> 255, flip if needed
}

AS_GRAY = True


files = sorted(glob(path.join(PATH_IN, "*.png")))
for f in files:
    
    # read as color image, perform bgr2rgb
    mask_in = cv.imread(f, cv.IMREAD_COLOR) 
    mask_in = cv.cvtColor(mask_in, cv.COLOR_BGR2RGB)

    # rgb to grayscale mapping 
    print('file:', f)
    mask_out = mask_in.copy()
    for k in mapping:
        print(k, '->', mapping[k])
        mask_out[(mask_out == k).all(axis = 2)] = mapping[k]
    
    # reduce channels to 1 if set
    if AS_GRAY:
        mask_out = mask_out[..., 0]
        
    mask_out = PIL.Image.fromarray(mask_out)
    mask_out.save(path.join(PATH_OUT, path.basename(f)))
