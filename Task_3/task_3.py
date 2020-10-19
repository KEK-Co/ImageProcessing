from time import time, sleep
from skimage.metrics import structural_similarity as ssim
from cv2 import imread, imshow, cvtColor, COLOR_BGR2HSV, COLOR_HSV2BGR, waitKey
from ImageProcessing.Task_1.psnr import psnr

PICTURES_TEST = ['test_picture.jpg']


def ratio(numer, denom):
    """Func to divide almost zero to almost zero"""
    numer_new = numer * 1000000 + 1
    denom_new = denom * 1000000 + 1
    return numer_new / denom_new


def convert_to_hsv(red, green, blue):
    """Algorithm to convert normalized RGB into normalized HSV"""
    MAX = max(red, green, blue)
    MIN = min(red, green, blue)
    # hue:
    if MAX == MIN:
        hue = 0
    elif MAX == red:
        hue = (60 * ((green - blue) / (MAX - MIN)) + 360) % 360
    elif MAX == green:
        hue = (60 * ((blue - red) / (MAX - MIN)) + 120) % 360
    elif MAX == blue:
        hue = (60 * ((red - green) / (MAX - MIN)) + 240) % 360
    else:
        raise ValueError("Unexpected result for hue")
    # saturation:
    if MAX == 0:
        satur = 0
    else:
        satur = 1 - MIN/MAX
    # value:
    value = MAX
    # scaling back
    hue = hue / 360
    return hue, satur, value


def convert_to_rgb(hue, satur, value):
    """Algorithm to convert normalized HSV into normalized RGB"""
    h = hue // 60 % 6
    v_min = ((100 - satur) * value) / 100
    a = (value - v_min) * (hue % 60) / 60
    v_inc = v_min + a
    v_dec = value - a
    h_dict = {0: (value, v_inc, v_min),
              1: (v_dec, value, v_min),
              2: (v_min, value, v_inc),
              3: (v_min, v_dec, value),
              4: (v_inc, v_min, value),
              5: (value, v_min, v_dec)}
    if h in h_dict.keys():
        return h_dict[h]
    raise ValueError('HSV to RGB conversion algorithm failed')


def pic_to_hsv(pic):
    """Func to convert RGB image into HSV image"""
    width = pic.shape[0]
    height = pic.shape[1]
    for x in range(width):
        for y in range(height):
            blue, green, red = pic[x, y] / 255
            h, s, v = convert_to_hsv(red, green, blue)
            pic[x, y] = (h * 255, s * 255, v * 255)


def pic_to_rgb(pic):
    width = pic.shape[0]
    height = pic.shape[1]
    for x in range(width):
        for y in range(height):
            hue, satur, value = pic[x, y] / 255
            r, g, b = convert_to_rgb(hue * 360, satur * 100, value * 100)
            pic[x, y] = (b * 2.55, g * 2.55, r * 2.55)


def brightness_RGB(pic, increment):
    for x in range(pic.shape[0]):
        for y in range(pic.shape[1]):
            blue, green, red = pic[x, y]
            red = min(255, max(0, red + increment))
            green = min(255, max(0, green + increment))
            blue = min(255, max(0, blue + increment))
            pic[x, y] = blue, green, red


def brightness_HSV(pic, increment):
    for x in range(pic.shape[0]):
        for y in range(pic.shape[1]):
            pic[x, y, 2] = min(max(pic[x, y, 2] + increment, 0), 255)


def test(pic_name):
    # before operations
    original = imread(pic_name)
    try:
        pic_handmade = original.copy()
        pic_cv2 = original.copy()
    except AttributeError:
        print('Error: incorrect picture name')
        return

    imshow('Source picture', original)

    # RGB -> HSV: hand-made implementation
    to_hsv_start = time()
    pic_to_hsv(pic_handmade)
    to_hsv_duration = time() - to_hsv_start
    imshow('RGB->HSV: hand-made implementation', pic_handmade)

    # RGB -> HSV: CV2 implementation
    to_hsv_cv2_start = time()
    pic_cv2 = cvtColor(pic_cv2, COLOR_BGR2HSV)
    to_hsv_cv2_duration = time() - to_hsv_cv2_start
    imshow('RGB->HSV: CV2 implementation', pic_cv2)

    print('\nRGB -> HSV: comparison:')
    print(f"   - Original and our implementation output PSNR similarity: {round(psnr(original, pic_handmade), 3)}")
    print(f"   - Original and CV2 implementation output PSNR similarity: {round(psnr(original, pic_cv2), 3)}")
    print(f"   - Implementation outputs SSIM comparison: {round(ssim(pic_handmade, pic_cv2, multichannel=True), 3)}")
    print(f"   - Time elapsed:")
    print(f"     Hand-made: {round(to_hsv_duration, 3)} sec; CV2: {round(to_hsv_cv2_duration, 3)} sec")
    print(f"     OpenCV implementation is almost {round(ratio(to_hsv_duration, to_hsv_cv2_duration))} times faster.\n")

    # HSV -> RGB: hand-made implementation
    to_rgb_start = time()
    pic_to_rgb(pic_handmade)
    to_rgb_duration = time() - to_rgb_start
    imshow('HSV->RGB: hand-made implementation',
           pic_handmade)

    # HSV -> RGB: CV2 implementation
    to_rgb_cv2_start = time()
    pic_cv2 = cvtColor(pic_cv2, COLOR_HSV2BGR)
    to_rgb_cv2_duration = time() - to_rgb_cv2_start
    imshow("HSV->RGB: CV2 implementation", pic_cv2)

    print('HSV -> RGB: comparison:')
    print(f"   - Original and our implementation output PSNR similarity: {round(psnr(original, pic_handmade), 3)}")
    print(f"   - Original and CV2 implementation output PSNR similarity: {round(psnr(original, pic_cv2), 3)}")
    print(f"   - Implementation outputs SSIM comparison: {round(ssim(pic_handmade, pic_cv2, multichannel=True), 3)}")
    print(f"   - Time elapsed:")
    print(f"     Hand-made: {round(to_rgb_duration, 3)} sec; CV2: {round(to_rgb_cv2_duration, 3)} sec")
    print(f"     OpenCV implementation is almost {round(ratio(to_rgb_duration, to_rgb_cv2_duration))} times faster.\n")

    brighter_rgb = original.copy()
    #brighter_hsv = original
    brighter_hsv = cvtColor(original.copy(), COLOR_BGR2HSV)

    INC = 50

    # Brightness +50: RGB
    br_rgb_start = time()
    brightness_RGB(brighter_rgb, INC * 2.55)
    br_rgb_duration = time() - br_rgb_start
    imshow('Brightness +50: RGB', brighter_rgb)

    # Brightness +50: HSV
    br_hsv_start = time()
    brightness_HSV(brighter_hsv, INC * 2.55)
    br_hsv_duration = time() - br_hsv_start
    brighter_hsv = cvtColor(brighter_hsv, COLOR_HSV2BGR)
    imshow('Brightness +50: HSV', brighter_hsv)

    print('Brightness +50:')
    print(f"   - Original and our implementation output PSNR similarity: {round(psnr(brighter_rgb, brighter_hsv), 3)}")
    print(f"   - Outputs SSIM comparison: {round(ssim(brighter_rgb, brighter_hsv, multichannel=True), 3)}")
    print(f"   - Time elapsed:")
    print(f"     RGB: {round(br_rgb_duration, 3)} sec; HSV: {round(br_hsv_duration, 3)} sec\n")

    print(f'---- TEST FOR {pic_name} FINISHED, PRESS ESC ---- ')
    waitKey()
    return


for item in PICTURES_TEST:
    test(item)
    sleep(3)
