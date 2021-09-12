import tkinter as tk
import tkinter.filedialog as tk_filedialog
import tkinter.messagebox as tk_messagebox

from time import sleep, time
from collections import defaultdict

import cv2
import numpy as np
from scipy.ndimage.filters import convolve, gaussian_filter
from tqdm import tqdm


# Task 1 - Settings
GAUSS_SIGMA = 1
SOBEL_AXIS = 1  # 0 = horizontal, 1 = vertical

# Task 2 - Settings
THRESHOLD = 0.3666  # from 0 to 1; for removing weakest pixels
THETA_NUM = 90  # Amount of directions; 360 / THETA_NUM must be int!
IGNORE_MINORITIES_R = 10  # Minimal circle radius we work with
SCALING_COEF = 15  # For postprocessing to work better on any image size
POST_PROCESSING = True  # Enable removing circles with close centers
POST_PROCESSING_THRESHOLD = 4  # For removing circles with close centers

# ======================================================================================================================
# All code before same separator with text "Task 4" is a release candidate, congratulations!
# ======================================================================================================================


def file_open():
    """Reading a file into cv2 image"""
    file = tk_filedialog.askopenfilename(title="Select image",
                                         filetypes=[("Image", ["jpg", "*.jpg", "*.png", "*.jpeg"])])
    while not file:
        raise FileNotFoundError

    # avoid errors with ru and other symbols - not use imread
    with open(file, 'rb') as stream:
        byte = bytearray(stream.read())
        array = np.asarray(byte, dtype=np.uint8)
        image = cv2.imdecode(array, cv2.IMREAD_UNCHANGED)

    if image is None:
        raise ValueError("Error: image for cv2.imread expected, got None")
    return image


def check_mode(mode):
    """For different output in task functions"""
    mode_list = frozenset(['default', 'silent', 'benchmarking'])
    if mode not in mode_list:
        raise NameError(f'''
        Trying to use mode {mode} which is not supported.
        Possible variants are:
        {mode_list}
        ''')
    return mode


# ======================================================================================================================
# Task 1 - Canny Algorithm
# ======================================================================================================================


def lab_1_gray_filter_UMNIE_LUDI(input_image):
    """Gray filter with Photoshop formula from lab 1"""
    blue = input_image[:, :, 0]
    green = input_image[:, :, 1]
    red = input_image[:, :, 2]
    output_image = (blue * 0.11 + green * 0.59 + red * 0.3)
    return output_image


def canny_sobel_filter(image):
    """Gradient searching"""
    x_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    y_kernel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    G_x = convolve(image, x_kernel)
    G_y = convolve(image, y_kernel)

    G = np.sqrt((np.square(G_x)) + (np.square(G_y)))
    theta = np.arctan2(G_y, G_x)
    return G, theta


def canny_non_maximum_sup(image, d):
    """Non-Maximum Suppression"""
    m, n = image.shape
    z = np.zeros((m, n), dtype=np.int32)
    angle = d * 180. / np.pi
    angle[angle < 0] += 180
    for i in range(1, m - 1):
        for j in range(1, n - 1):
            q = 255
            r = 255
            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                q = image[i, j + 1]
                r = image[i, j - 1]
            elif 22.5 <= angle[i, j] < 67.5:
                q = image[i + 1, j - 1]
                r = image[i - 1, j + 1]
            elif 67.5 <= angle[i, j] < 112.5:
                q = image[i + 1, j]
                r = image[i - 1, j]
            elif 112.5 <= angle[i, j] < 157.5:
                q = image[i - 1, j - 1]
                r = image[i + 1, j + 1]
            if image[i, j] >= q and image[i, j] >= r:
                z[i, j] = image[i, j]
            else:
                z[i, j] = 0
    return z


def canny_thresholding(image):
    """Detecting weak & strong pixels"""
    # default settings
    strong = 1.0
    weak = 0.5
    low_border_coef = 0.1
    high_border_coef = 0.4

    results = np.zeros(image.shape)
    max_pix = image.max()
    low, high = low_border_coef * max_pix, high_border_coef * max_pix
    weaks = []
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            px = image[i][j]
            if px >= high:
                results[i][j] = strong
            elif px >= low:
                results[i][j] = weak
                weaks.append((i, j))
    return results, weaks


def canny_tracing(image, weaks):
    """Supressing weak pixels not connected to strong ones"""
    # default settings
    STRONG = 1.0
    WEAK = 0.5

    for item in weaks:
        i, j = item
        if image[i, j] == WEAK:
            if ((image[i + 1, j - 1] == STRONG) or (image[i + 1, j] == STRONG) or (image[i + 1, j + 1] == STRONG)
                    or (image[i, j - 1] == STRONG) or (image[i, j + 1] == STRONG) or (image[i - 1, j - 1] == STRONG)
                    or (image[i - 1, j] == STRONG) or (image[i - 1, j + 1] == STRONG)):
                image[i, j] = STRONG
            else:
                image[i, j] = 0
    return image


def canny_main(input_image=None, gauss_sigma=GAUSS_SIGMA, mode='default'):
    """Steps for Canny algorithm"""
    if input_image is None:
        try:
            image = file_open()
        except FileNotFoundError:
            return
    else:
        image = input_image.copy()

    check_mode(mode)
    if mode == 'default':
        cv2.imshow("Image", image)
        cv2.waitKey(30)

        print("\nCanny - Applying gray filter...")
        # gray_img = lab_1_gray_filter(image)
        gray_img = lab_1_gray_filter_UMNIE_LUDI(image)
        print("=== Done === \n")

        print("Canny - Applying Gaussian filter...")
        gaussian_img = gaussian_filter(gray_img, gauss_sigma)
        print("=== Done === \n")

        print("Canny - Processing gradient searching...")
        sobel_img, theta = canny_sobel_filter(gaussian_img)
        print("=== Done === \n")

        print("Canny - Processing non-maximum supression...")
        non_max_img = canny_non_maximum_sup(sobel_img, theta)
        print("=== Done === \n")

        print("Canny - Processing double thresholding...")
        results, weaks = canny_thresholding(non_max_img)
        print("=== Done === \n")

        print("Canny - Tracing...")
        canny_img = canny_tracing(results, weaks)
        print("=== Done === \n")

        cv2.imshow("Canny's Algorithm", canny_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    elif mode == 'silent' or 'benchmarking':
        # muted steps which are used in Hough algorithm not to overload output
        gray_img = lab_1_gray_filter_UMNIE_LUDI(image)
        gaussian_img = gaussian_filter(gray_img, gauss_sigma)
        sobel_img, theta = canny_sobel_filter(gaussian_img)
        non_max_img = canny_non_maximum_sup(sobel_img, theta)
        results, weaks = canny_thresholding(non_max_img)
        canny_img = canny_tracing(results, weaks)
    else:
        raise ValueError('Mode is not chosen correctly in function canny_main')

    return canny_img


# ======================================================================================================================
# Task 2 - Hough Transform
# ======================================================================================================================


def hough_preparation(img_width, img_height, theta_num=THETA_NUM, r_min=IGNORE_MINORITIES_R):
    """Pre-count sin and cos values and get combinations"""
    r_max = int(min(img_height, img_width) / 2)
    radiuses = [i for i in range(r_min, r_max)]
    thetas = np.arange(0, 360, step=int(360/theta_num))

    cos_thetas = np.cos(np.deg2rad(thetas))
    sin_thetas = np.sin(np.deg2rad(thetas))

    combinations = []
    for r in radiuses:
        for t in range(theta_num):
            combinations.append((r, int(r * cos_thetas[t]), int(r * sin_thetas[t])))

    combinations = sorted(combinations, key=lambda i: i[0])
    return combinations


def hough_accumulate(img_width, img_height, image_canny, combinations, bench=False):
    """Voting for candidate circles passing through pixel"""
    accumulator = defaultdict(int)

    if not bench:
        sleep(0.2)  # for tqdm sync
        for y in tqdm(range(img_height)):
            for x in range(img_width):
                if image_canny[y][x] != 0:  # white pixel
                    for r, rcos_t, rsin_t in combinations:
                        x_center = x - rcos_t
                        y_center = y - rsin_t
                        accumulator[(x_center, y_center, r)] += 1  # vote for current candidates
        sleep(0.2)  # for tqdm sync
    else:
        for y in range(img_height):
            for x in range(img_width):
                if image_canny[y][x] != 0:  # white pixel
                    for r, rcos_t, rsin_t in combinations:
                        x_center = x - rcos_t
                        y_center = y - rsin_t
                        accumulator[(x_center, y_center, r)] += 1  # vote for current candidates
    return accumulator


def hough_detect_circles(accumulator, theta_num=THETA_NUM, threshold=THRESHOLD, bench=False):
    """ Getting output list of detected circles"""
    out_circles = []

    for candidate_circle, votes in sorted(accumulator.items(), key=lambda i: -i[1]):
        x, y, r = candidate_circle
        current_vote_percentage = votes / theta_num
        if current_vote_percentage >= threshold:
            out_circles.append((x, y, r, current_vote_percentage))

    if not bench:
        print(f"Circles found: {len(out_circles)}")
    return out_circles


def hough_post_processing(out_circles,
                          post_processing_threshold=POST_PROCESSING_THRESHOLD, bench=False):
    """Removing circles which have too close centers"""

    def check_dot(x, y, r, v):
        for xc, yc, rc, vv in postprocess_circles:
            if not (abs(x - xc) > post_processing_threshold
                    or abs(y - yc) > post_processing_threshold
                    or abs(r - rc) > post_processing_threshold):
                return
        postprocess_circles.append((x, y, r, v))

    postprocess_circles = []

    if not bench:
        sleep(0.2)  # for tqdm sync
        for x, y, r, v in tqdm(out_circles):
            check_dot(x, y, r, v)
        out_circles = postprocess_circles
        sleep(0.2)  # for tqdm sync
        print(f"Circles found after postprocessing: {len(out_circles)}")
    else:
        for x, y, r, v in out_circles:
            check_dot(x, y, r, v)
        out_circles = postprocess_circles
    return out_circles


def hough_draw(input_image, circles_list):
    """Highlighting the result"""
    line_width = 2
    output_image = input_image.copy()
    for x, y, r, v in circles_list:
        output_image = cv2.circle(output_image, (x, y), r, (0, 255, 0), line_width)
    return output_image


def hough_main(input_image=None, post_processing=POST_PROCESSING, theta_num=THETA_NUM,
               threshold=THRESHOLD, post_processing_threshold=POST_PROCESSING_THRESHOLD,
               r_min=IGNORE_MINORITIES_R, mode='default'):
    """Steps for Hough transform"""
    if input_image is None:
        try:
            image = file_open()
        except FileNotFoundError:
            return
    else:
        image = input_image.copy()
    cv2.imshow("Image", image)
    img_height, img_width = image.shape[:2]

    check_mode(mode)
    if mode == 'default':
        print()
        print("Hough - Calculating Canny algorithm...")
        canny_img = canny_main(image, mode='silent')
        cv2.imshow("Canny's Algorithm", canny_img)
        print("=== Done === \n")
        cv2.waitKey(30)

        print("Hough - Calculating sin/cos for Hough algorithm...")
        combinations = hough_preparation(img_width, img_height, theta_num, r_min)
        print("=== Done === \n")

        print("Hough - Accumulating votes...")
        accumulator = hough_accumulate(img_width, img_height, canny_img, combinations)
        print("=== Done === \n")

        print("Hough - Detecting circles...")
        circles_list = hough_detect_circles(accumulator, theta_num, threshold)
        print("=== Done === \n")

        if post_processing:
            print("Hough - Postprocessing for close circles...")
            circles_list = hough_post_processing(circles_list, post_processing_threshold)
            print("=== Done === \n")

        print("Hough - Drawing circles...")
        hough_img = hough_draw(image, circles_list)
        print("=== Done === \n")

        cv2.imshow("Hough Detection", hough_img)
        cv2.waitKey(0)
        cv2.destroyWindow("Canny's Algorithm")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    elif mode == 'silent':
        canny_img = canny_main(image, mode='silent')
        combinations = hough_preparation(img_width, img_height, theta_num, r_min)
        accumulator = hough_accumulate(img_width, img_height, canny_img, combinations)
        circles_list = hough_detect_circles(accumulator, theta_num, threshold)
        if post_processing:
            circles_list = hough_post_processing(circles_list, post_processing_threshold)
        hough_img = hough_draw(image, circles_list)

    elif mode == 'benchmarking':
        canny_img = canny_main(image, mode='benchmarking')
        combinations = hough_preparation(img_width, img_height, theta_num, r_min)
        accumulator = hough_accumulate(img_width, img_height, canny_img, combinations, bench=True)
        circles_list = hough_detect_circles(accumulator, theta_num, threshold, bench=True)
        if post_processing:
            circles_list = hough_post_processing(circles_list, post_processing_threshold, bench=True)
        hough_img = hough_draw(image, circles_list)

    else:
        raise ValueError('Mode is not chosen correctly in function hough_main')

    return hough_img


# ======================================================================================================================
# Function for Hough Transform from CV2
# ======================================================================================================================


def hough_cv2(input_image=None, post_processing_threshold=POST_PROCESSING_THRESHOLD,
              r_min=IGNORE_MINORITIES_R, mode='default'):
    """Wrapper for algorithm from CV2 with our parameters"""
    # Documentation:
    # https://www.pyimagesearch.com/2014/07/21/detecting-circles-images-using-opencv-hough-circles/

    if input_image is None:
        try:
            image = file_open()
        except FileNotFoundError:
            return
    else:
        image = input_image.copy()

    grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grey_image = cv2.medianBlur(grey_image, 5)

    circles = cv2.HoughCircles(grey_image, cv2.HOUGH_GRADIENT,
                               1,  # the inverse ratio of the accumulator resolution to the image resolution
                               55,  # min distance between the centers of detected circles, ~POST_PROCESSING_THRESHOLD
                               param1=50,  # Gradient value used to handle edge detection in the Yuen et al. method
                               param2=30,  # Accumulator threshold value
                               minRadius=r_min)  # Minimum size of the radius (in pixels).

    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)

    if mode == 'default':
        cv2.imshow("CV2 Hough Image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return image


# ======================================================================================================================
# Task 4 - Compare implementations
# ======================================================================================================================


def compare(input_image=None):
    """Algorithm for benchmarking"""
    if input_image is None:
        try:
            image = file_open()
        except FileNotFoundError:
            return
    else:
        image = input_image.copy()
    cv2.imshow("Image", image)

    # Canny
    print('CV2 Canny: started...')
    start_time = time()
    cv2_canny = cv2.Canny(image, 75, 255)  # For better result need to convert to our 0.1 and 0.4 thresholds?
    cv2_canny_time = time() - start_time
    print(f'=== CV2 Canny: Done in {cv2_canny_time}s === \n')

    print('Our Canny: started...')
    start_time = time()
    my_canny = canny_main(image, mode='benchmarking')
    my_canny_time = time() - start_time
    print(f'=== Our Canny: Done in {my_canny_time}s === \n')

    # Hough
    print('CV2 Hough: started...')
    start_time = time()
    cv2_hough = hough_cv2(image, mode='benchmarking')
    cv2_hough_time = time() - start_time
    print(f'=== CV2 Hough: Done in {cv2_hough_time}s === \n')

    print('Our Hough: started...')
    start_time = time()
    my_hough = hough_main(image, mode='benchmarking')
    my_hough_time = time() - start_time
    print(f'=== Our Hough: Done in {my_hough_time}s === \n')

    sleep(0.2)

    cv2.imshow('CV2 Canny Image', cv2_canny)
    cv2.imshow('Our Canny Image', my_canny)
    cv2.imshow('CV2 Hough Image', cv2_hough)
    cv2.imshow('Our Hough Image', my_hough)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return cv2_canny_time, my_canny_time, cv2_hough_time, my_hough_time


def compare_main(input_image=None):
    """Graphic interface for compare function"""
    if input_image is None:
        try:
            image = file_open()
        except FileNotFoundError:
            return
    else:
        image = input_image.copy()

    window = tk.Tk()
    window.title("Benchmarking")

    tk_messagebox.showinfo("Warning", "Please be ready to wait up to 2 minutes while benchmarking. \n"
                           "Hough algorithm in our implementation takes some time.")

    cv2_canny_time, my_canny_time, cv2_hough_time, my_hough_time = compare(image)

    # Formatting
    tk.Label(window, text="Time elapsed").grid(row=0, column=0, columnspan=5)
    tk.Label(window, text="Algorithm").grid(row=1, column=0)
    tk.Label(window, text="KEK Corp. \nimplementation").grid(row=1, column=2)
    tk.Label(window, text="CV2\n implementation").grid(row=1, column=4)

    for row in range(1, 4):  # spreadsheet borders
        for column in range(1, 5, 2):
            tk.Label(window, text="|\n|").grid(row=row, column=column)

    # Canny results
    tk.Label(window, text="Canny Alg").grid(row=2, column=0)
    tk.Label(window, text=str(round(my_canny_time, 4))).grid(row=2, column=2)
    tk.Label(window, text=str(round(cv2_canny_time, 4))).grid(row=2, column=4)

    # Hough results
    tk.Label(window, text="Hough Alg").grid(row=3, column=0)
    tk.Label(window, text=str(round(my_hough_time, 4))).grid(row=3, column=2)
    tk.Label(window, text=str(round(cv2_hough_time, 4))).grid(row=3, column=4)

    # Put report on top
    window.attributes('-topmost', True)
    window.update()
    window.attributes('-topmost', False)

    window.bind("<Escape>", lambda event: window.destroy())  # tkinter needs a function to bind


# ======================================================================================================================
# Main interface
# ======================================================================================================================


def run_graphic_interface():
    """Buttons and forms"""
    main_frame = tk.Tk()
    main_frame.title("Лабораторная 3")

    def close():
        main_frame.quit()

    # Canny algorithm button
    button_canny = tk.Button(main_frame, text='Canny algorithm', activebackground="#498200",
                             command=canny_main, height=2, width=40)
    button_canny.grid(row=0, column=0, columnspan=2, sticky=tk.N+tk.S+tk.W+tk.E)

    # Hough algorithm button
    button_hough = tk.Button(main_frame, text='Hough algorithm', activebackground="#498200",
                             command=hough_main, height=2, width=40)
    button_hough.grid(row=1, column=0, columnspan=2, sticky=tk.N+tk.S+tk.W+tk.E)

    # Compare Ours to CV2 button
    button_compare = tk.Button(main_frame, text='Compare Ours to CV2', activebackground="#498200",
                               command=compare_main, height=2, width=25)
    button_compare.grid(row=2, column=0, sticky=tk.N+tk.S+tk.W+tk.E)

    # Test CV2 Hough button
    button_hough_cv2 = tk.Button(main_frame, text='Test CV2 Hough', activebackground="#498200",
                                 command=hough_cv2, height=2, width=15)
    button_hough_cv2.grid(row=2, column=1, sticky=tk.N+tk.S+tk.W+tk.E)

    # Test CV2 Hough button
    button_exit = tk.Button(main_frame, text='Exit', background="#498200", activebackground="#900020",
                            command=close, height=2, width=10)
    button_exit.grid(row=4, column=0, columnspan=2, sticky=tk.N + tk.S + tk.W + tk.E)

    main_frame.mainloop()


if __name__ == "__main__":
    run_graphic_interface()
