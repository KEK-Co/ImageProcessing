from cv2 import cv2
import numpy as np
from PIL import Image
from numpy import *


def pool(input):
    output = np.empty(
        shape=(int(input.shape[0] / 2), int(input.shape[1] / 2), input.shape[2])
    )
    for h in range(output.shape[0]):
        for w in range(output.shape[1]):
            output[h, w] = np.max(
                input[h * 2: (h + 1) * 2, w * 2: (w + 1) * 2],
                axis=(0, 1),
            )
    return output


def conv(input):
    filters = np.random.normal(size=(3, 3, 3, 5))
    f_h, f_w, f_c, f_num = filters.shape
    i_h, i_w, i_c = input.shape

    o_h = int((i_h - f_h))
    o_w = int((i_w - f_w))

    output = np.zeros(shape=(o_h, o_w, f_num))

    for i in range(f_num):
        for h in range(o_h):
            for w in range(o_w):
                res = 0
                for y in range(f_h):
                    for x in range(f_w):
                        for c in range(f_c):
                            res += (input[h + y][w + x][c] * filters[y][x][c][i])
                output[h][w][i] = res
    return output


def softmax(input):
    i_h, i_w, _ = input.shape
    output = np.zeros(shape=input.shape)
    for h in range(i_h):
        for w in range(i_w):
            output[h, w, :] = np.exp(input[h, w]) / sum(np.exp(input)[h, w])
    return output


if __name__ == "__main__":
    image = cv2.imread("test_image_2.jpg")

    output_1 = conv(image)
    print("conv:", output_1.shape)
    img = Image.fromarray(output_1, 'RGB')
    img.show()

    output_2 = output_2 = (output_1 - output_1.mean()) / output_1.std()
    print("norm:", output_2.shape)
    img = Image.fromarray(output_2, 'RGB')
    img.show()

    output_3 = np.maximum(output_2, 0)
    print("relu:", output_3.shape)
    img = Image.fromarray(output_3, 'RGB')
    img.show()

    output_4 = pool(output_3)
    print("pool:", output_4.shape)
    img = Image.fromarray(output_4, 'RGB')
    img.show()

    output_5 = softmax(output_4)
    print("softMax:", output_5.shape)
    img = Image.fromarray(output_5, 'RGB')
    img.show()

    cv2.imshow('image', image)
    cv2.waitKey()
