import cv2
import math
import numpy as np


def psnr(first, second):
    if first.shape != second.shape:
        print('Image size mismatch')
        return 0
    first = first.astype(np.float64) / 255.
    second = second.astype(np.float64) / 255.
    mse = np.mean((first - second) ** 2)
    if mse == 0:
        return 100
    return 10 * math.log10(1.0 ** 2 / mse)


img = cv2.imread("joka.jpg", 1)
img2 = cv2.imread("joka_feat_noize.jpg", 1)
print(f'CV2 gray: {cv2.PSNR(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY),255)}')
print(f'Pair similarity gray: {psnr(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY))}')
print(f'CV2: {cv2.PSNR(img, img2,255)}')
print(f'Pair similarity: {psnr(img, img2)}')
cv2.imshow("Original", img)
cv2.imshow("Compressed", img)
cv2.waitKey()