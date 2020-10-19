import math
import cv2
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
    return 10 * math.log10(1.0 / mse)


if __name__ == '__main__':
    img = cv2.imread("joka.jpg", 1)
    img2 = cv2.imread("joka_feat_noize.jpg", 1)
    img_gray, img2_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),\
                          cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    print(f'Hand-made PSNR:\n {psnr(img, img2)}')
    print(f'CV2 built-in PSNR:\n {cv2.PSNR(img, img2,255)}\n')
    print(f'Hand-made PSNR for grayscale:\n {psnr(img_gray, img2_gray)}')
    print(f'CV2 built-in PSNR for grayscale:\n {cv2.PSNR(img_gray, img2_gray,255)}\n')

    cv2.imshow("Original", img)
    cv2.imshow("Compressed", img2)
    cv2.waitKey()
