import cv2
import numpy as np

if __name__ == "__main__":

    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    img = cv2.imread("Lena.png")
    img_norect = cv2.imread("Lena.png")
    cv2.imshow('original', img)

    faces = face_cascade.detectMultiScale(img, 1.1, 1)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 5)
    cv2.imshow('find face', img)

    for (x, y, w, h) in faces:
        img = img_norect[int(y * 0.9): (y + int(h * 1.1)), int(x * 0.9): (x + int(w * 1.1))]
    cv2.imshow('cropped', img)

    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_canny = cv2.Canny(img_grey, 100, 100, 3)
    cv2.imshow('canny', img_canny)

    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img_canny, 8)
    edged = np.zeros(img_grey.shape, np.uint8)
    for i in range(1, nb_components):
        if stats[i, cv2.CC_STAT_WIDTH] < 10 and stats[i, cv2.CC_STAT_HEIGHT] < 10:
            continue
        componentMask = (output == i).astype(np.uint8) * 255
        edged = cv2.bitwise_or(edged, componentMask)
    cv2.imshow('edgesfiltered', edged)

    kernel = np.ones((5, 5), np.uint8)
    img_dilated = cv2.dilate(edged, kernel)
    cv2.imshow('dilated', img_dilated)

    M = cv2.GaussianBlur(img_dilated, (5, 5), 5, 5)
    M = M / 255
    cv2.imshow('M', M)

    F1 = cv2.bilateralFilter(img, 5, 75, 75)
    cv2.imshow('F1', F1)

    F2 = cv2.addWeighted(img, 1.5, cv2.GaussianBlur(img, (9, 9), 10.0), -0.5, 0)
    cv2.imshow('F2', F2)

    M = M[:, :, np.newaxis]
    result = M * F2 + (1 - M) * F1
    result = result.astype(np.uint8)

    cv2.imshow('result', result)
    cv2.waitKey(0)
