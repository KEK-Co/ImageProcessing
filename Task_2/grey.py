import cv2
import time

img = cv2.imread("rick-roll.jpg")
img_cv = img.copy()
img_av = img.copy()
cv2.imshow('Original', img)

st = time.time()
img_cv = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
end = time.time()
print('OpenCV: ', end - st)
cv2.imshow('OpenCV', img_cv)

st = time.time()
blue = img[:, :, 0]
green = img[:, :, 1]
red = img[:, :, 2]
gray = (0.11 * blue + 0.59 * green + 0.3 * red)
img_av[:, :, 0] = gray
img_av[:, :, 1] = gray
img_av[:, :, 2] = gray
end = time.time()

print('Average: ', end - st)
cv2.imshow('Photoshop', img_av)
cv2.waitKey(666)