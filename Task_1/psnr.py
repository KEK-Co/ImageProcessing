import cv2
import math
import numpy as np


def psnr(first, second):
    if first.shape != second.shape:
        print('Image size mismatch')
        return 0
    mse = np.mean((first - second) ** 2)
    if mse == 0:
        return 100
    max_pix = 255
    return 10 * math.log10(max_pix ** 2 / mse)


img = cv2.imread("boka.jpg", 1)
img2 = cv2.imread("joka.jpg", 1)
print(psnr(img, img2))
