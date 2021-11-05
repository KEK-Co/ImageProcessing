from cv2 import cv2, imread, imshow, waitKey, IMREAD_GRAYSCALE
import time
import numpy as np
import math
import matplotlib.pyplot as plt
import random
import tkinter as tk
import os

def get_distribution(sigma, nu, number):
    values = np.linspace(0, 255, number)
    probabilities = 1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-np.power(values - nu, 2) / (2 * np.power(sigma, 2)))
    probabilities_sum = np.cumsum(probabilities * (255 - 0) / number)
    return np.stack((probabilities_sum, values), axis=-1)

def noise_gen_gauss(img: np.array, sigma, nu, percentage = 0.05):
    # Преобразуем к Python list из-за соображений производительности итерации
    img_list = img.tolist()
    number_of_noise_pixels = int(len(img_list) * len(img_list[0]) * percentage)
    distribution = get_distribution(sigma, nu, 2000)

    for i in range(number_of_noise_pixels):
        index = int(random.uniform(0, len(img_list) * len(img_list[0])))
        random_uniform_value = random.uniform(0, 1)

        distribution_index = distribution[:,0].searchsorted(random_uniform_value * distribution[-1, 0])

        if distribution_index >= 2000:
            random_value = 255
        else:
            random_value = round(distribution[distribution_index, 1])

        img_list[index // img.shape[1]][index % img.shape[1]] += random_value

    return np.clip(img_list, 0, 255)

def PSNR(im1, im2):
    mse = np.mean((im1 - im2) ** 2)
    if (mse == 0):
        return 100
    max_pixel = 255.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr

def clamp(val, max, min):
    if(val <= min):
        return min
    elif(val >= max):
        return max
    else:
        return val

def spFilter(image,koef):
    result = np.zeros(image.shape,np.uint8)
    secondKoef = 1 - koef
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rand = random.random()
            if rand < koef:
                result[i][j] = 0
            elif rand > secondKoef:
                result[i][j] = 255
            else:
                result[i][j] = image[i][j]
    return result

def medianFilter(imageOrig):
    image=imageOrig
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rad = 3
            n = rad*rad
            radius = rad//2
            arrR=[]
            arrG=[]
            arrB=[]
            for k in range(n):
                arrR.append(0)
                arrG.append(0)
                arrB.append(0)
            k=0
            for x in range(-radius,radius+1):
                for y in range(-radius,radius+1):
                    ix = clamp(i+x,image.shape[0]-1,0)
                    jy = clamp(j+y,image.shape[1]-1,0)
                    (b, g, r) = image[ix,jy]
                    arrR[k]=r
                    arrG[k]=g
                    arrB[k]=b
                    k+=1
            arrR.sort()
            arrG.sort()
            arrB.sort()
            image[i,j] = (arrB[n//2],arrG[n//2],arrR[n//2])
    return image

def averageFilter(imageOrig):
    image=imageOrig
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rad = 5
            n = rad*rad
            radius = rad//2
            mR=0
            mG=0
            mB=0
            for x in range(-radius,radius+1):
                for y in range(-radius,radius+1):
                    ix = clamp(i+x,image.shape[0]-1,0)
                    jy = clamp(j+y,image.shape[1]-1,0)
                    (b, g, r) = image[ix,jy]
                    mR+=r
                    mG+=g
                    mB+=b
            image[i,j] = (mB//n,mG//n,mR//n)
    return image

def visualTestFilters():
    image = cv2.imread("test_image.jpg")
    PICTURES_TEST = ['test_image.jpg', 'test_image_2.jpg']
    for item in PICTURES_TEST:
        if not os.path.exists(item):
            raise ValueError
        else:
            original = imread(item, IMREAD_GRAYSCALE)
            imshow('Source picture', original)

            img_test1 = noise_gen_gauss(original, 63.75, 127.5, 0.05)
            plt.imshow(np.uint8(img_test1))
            plt.show()

            img_test2 = noise_gen_gauss(original, 63.75, 127.5, 0.2)
            plt.imshow(np.uint8(img_test2))
            plt.show()
            waitKey()
    cv2.imshow("Original", image)
    sp = spFilter(image, 0.1)
    cv2.imshow("S&P", sp)
    median = medianFilter(sp)
    cv2.imshow("median", median)
    average = averageFilter(sp)
    cv2.imshow("average", average)
    medianCV = cv2.medianBlur(sp, 3)
    cv2.imshow("cvmedian", medianCV)

    cv2.waitKey(0)

def compareMineVsCV():
    image = cv2.imread("test_image.jpg")
    """Create window with results"""
    windowResults = tk.Toplevel(root)
    lbMy=tk.Label(windowResults, text="||\n||\n||\n||\n||").grid(row = 0, column = 1,rowspan = 5)
    lbMy=tk.Label(windowResults, text="Time").grid(row = 0, column = 2)
    lbMy=tk.Label(windowResults, text="||\n||\n||\n||\n||").grid(row = 0, column = 3,rowspan = 5)
    lbMy=tk.Label(windowResults, text="Similarity").grid(row = 0, column = 4)
    lbMy=tk.Label(windowResults, text="Salt & Pepper realization").grid(row = 1, column = 0)
    lbMy=tk.Label(windowResults, text="Median realization").grid(row = 2, column = 0)
    lbCV=tk.Label(windowResults, text="Average realization").grid(row = 3, column = 0)
    lbCV=tk.Label(windowResults, text="CV median realization").grid(row = 4, column = 0)

    startTime = time.time()
    sp = spFilter(image, 0.1)
    myResultTimeSP = time.time() - startTime
    mySPSimilarity = PSNR(image, sp)

    startTime = time.time()
    medianCV = cv2.medianBlur(sp, 3)
    cvResultTime= time.time() - startTime
    cvMedianSimilarity = PSNR(image, medianCV)

    startTime = time.time()
    median = medianFilter(image)
    myResultTimeMedian = time.time() - startTime
    myMedianSimilarity = PSNR(medianCV, median)

    startTime = time.time()
    average = averageFilter(image)
    myResultTimeAverage = time.time() - startTime
    myAverageSimilarity = PSNR(medianCV, average)

    """Change window with results"""
    lbMy=tk.Label(windowResults, text=str(round(myResultTimeSP,4))).grid(row = 1, column = 2)
    lbMy=tk.Label(windowResults, text=str(mySPSimilarity)+' %').grid(row = 1, column = 4)
    lbMy=tk.Label(windowResults, text=str(round(myResultTimeMedian,4))).grid(row = 2, column = 2)
    lbMy=tk.Label(windowResults, text=str(myMedianSimilarity)+' %').grid(row = 2, column = 4)
    lbMy=tk.Label(windowResults, text=str(round(myResultTimeAverage,4))).grid(row = 3, column = 2)
    lbMy=tk.Label(windowResults, text=str(myAverageSimilarity)+' %').grid(row = 3, column = 4)
    lbMy=tk.Label(windowResults, text=str(round(cvResultTime,4))).grid(row = 4, column = 2)
    lbMy=tk.Label(windowResults, text=str(cvMedianSimilarity)+' %').grid(row = 4, column = 4)


root = tk.Tk()
hsvBut1 = tk.Button(root, text = 'Visual test of filters', activebackground = "#555555", command = visualTestFilters).grid(row = 0, column = 0)
hsvBut2 = tk.Button(root, text = 'Comparison of my and cv realization', activebackground = "#555555", command = compareMineVsCV).grid(row = 0, column = 2)
lbFreeSpace = tk.Label(root, text = '||').grid(row = 0, column = 1)
root.mainloop()

