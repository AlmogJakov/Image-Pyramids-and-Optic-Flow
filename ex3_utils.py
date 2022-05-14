import sys
from typing import List

import numpy as np
import cv2
from numpy.linalg import LinAlgError
import matplotlib.pyplot as plt


def myID() -> np.int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 203201389


# ---------------------------------------------------------------------------
# ------------------------ Lucas Kanade optical flow ------------------------
# ---------------------------------------------------------------------------

"""
Since A^T * A is 2X2 Matrix we can get at most 2 eigenvalues.
 * if eigenvalues are small (e.g smaller then 1) there is no gradient in all directions, 
      it is likely we are looking at pixels over a flat area that don’t show any correlation.
 * if eigenvalues show a large ratio (one strong, one weak gradient) it means we’re looking at an edge.
Source: https://medium.com/building-autonomous-flight-software/lucas-kanade-optical-flow-942d6bc5a078
"""


# https://sandipanweb.wordpress.com/2018/02/25/implementing-lucas-kanade-optical-flow-algorithm-in-python/
def opticalFlow(im1: np.ndarray, im2: np.ndarray, step_size=10,
                win_size=5) -> (np.ndarray, np.ndarray):
    """
    Given two images, returns the Translation from im1 to im2
    :param im1: Image 1
    :param im2: Image 2
    :param step_size: The image sample size
    :param win_size: The optical flow window size (odd number)
    :return: Original points [[x,y]...], [[dU,dV]...] for each points
    """
    img1, img2 = im1, im2
    if len(im1.shape) == 3:
        img1 = cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY)
    if len(im2.shape) == 3:
        img2 = cv2.cvtColor(im2, cv2.COLOR_RGB2GRAY)
    I_x, I_y = __get_directions(img1)
    I_t = np.subtract(img1, img2)
    # kernel_x = np.array([[-1., 1.], [-1., 1.]])
    # kernel_y = np.array([[-1., -1.], [1., 1.]])
    # kernel_t = np.array([[1., 1.], [1., 1.]]) * .25  # *.25
    # I_x = cv2.filter2D(img1, -1, kernel_x)
    # I_y = cv2.filter2D(img1, -1, kernel_y)
    # I_t = cv2.filter2D(img2, -1, kernel_t) + cv2.filter2D(img1, -1, -kernel_t)
    height, width = img2.shape
    half_win_size, num_of_win_pixels = win_size // 2, win_size ** 2
    u_v_list, y_x_list = [], []
    for i in range(int(max(step_size, win_size) / 2), height - int(max(step_size, win_size) / 2), step_size):
        for j in range(int(max(step_size, win_size) / 2), width - int(max(step_size, win_size) / 2), step_size):
            x_win = I_x[i - half_win_size: i + half_win_size + 1, j - half_win_size: j + half_win_size + 1]
            y_win = I_y[i - half_win_size: i + half_win_size + 1, j - half_win_size: j + half_win_size + 1]
            A = np.hstack((x_win.reshape(num_of_win_pixels, 1), y_win.reshape(num_of_win_pixels, 1)))
            if not __acceptable_eigenvalues(A):
                continue
            t_win = I_t[i - half_win_size: i + half_win_size + 1, j - half_win_size: j + half_win_size + 1]
            b = (-1) * t_win.reshape(num_of_win_pixels, 1)
            y_x_list.append((j, i))
            u_v = np.dot(np.linalg.pinv(A), b)
            u_v_list.append(u_v)
    return np.array(y_x_list).reshape(-1, 2), np.array(u_v_list).reshape(-1, 2)


MIN_LAMDA = 1
MAX_LAMDA_RATIO = 50


def __acceptable_eigenvalues(A: np.ndarray) -> bool:
    squared_matrix = np.dot(np.transpose(A), A)
    lamda1, lamda2 = np.sort(np.linalg.eigvals(squared_matrix))
    if lamda1 > MIN_LAMDA and lamda2 > lamda1 and (lamda2 / lamda1) < MAX_LAMDA_RATIO:
        return True
    return False


def __get_directions(img: np.ndarray):
    v = np.array([[1, 0, -1]])
    X = cv2.filter2D(img, -1, v)
    Y = cv2.filter2D(img, -1, v.T)
    return X, Y


def opticalFlowP(im1: np.ndarray, im2: np.ndarray, step_size=10,
                 win_size=5) -> (np.ndarray, np.ndarray):
    """
    Given two images, returns the Translation from im1 to im2
    :param im1: Image 1
    :param im2: Image 2
    :param step_size: The image sample size
    :param win_size: The optical flow window size (odd number)
    :return: Original points [[x,y]...], [[dU,dV]...] for each points
    """
    img1, img2 = im1, im2
    if len(im1.shape) == 3:
        img1 = cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY)
    if len(im2.shape) == 3:
        img2 = cv2.cvtColor(im2, cv2.COLOR_RGB2GRAY)
    I_x, I_y = __get_directions(img1)
    I_t = np.subtract(img1, img2)
    # kernel_x = np.array([[-1., 1.], [-1., 1.]])
    # kernel_y = np.array([[-1., -1.], [1., 1.]])
    # kernel_t = np.array([[1., 1.], [1., 1.]]) * .25  # *.25
    # I_x = cv2.filter2D(img1, -1, kernel_x)
    # I_y = cv2.filter2D(img1, -1, kernel_y)
    # I_t = cv2.filter2D(img2, -1, kernel_t) + cv2.filter2D(img1, -1, -kernel_t)
    height, width = img2.shape
    half_win_size, num_of_win_pixels = win_size // 2, win_size ** 2
    u_v_list, y_x_list = [], []
    u = np.zeros(img1.shape)
    v = np.zeros(img1.shape)
    for i in range(int(max(step_size, win_size) / 2), height - int(max(step_size, win_size) / 2), step_size):
        for j in range(int(max(step_size, win_size) / 2), width - int(max(step_size, win_size) / 2), step_size):
            x_win = I_x[i - half_win_size: i + half_win_size + 1, j - half_win_size: j + half_win_size + 1]
            y_win = I_y[i - half_win_size: i + half_win_size + 1, j - half_win_size: j + half_win_size + 1]
            print(x_win.shape)
            A = np.hstack((x_win.reshape(num_of_win_pixels, 1), y_win.reshape(num_of_win_pixels, 1)))
            y_x_list.append((j, i))
            if not __acceptable_eigenvalues(A):
                u_v_list.append((0, 0))
                continue
            t_win = I_t[i - half_win_size: i + half_win_size + 1, j - half_win_size: j + half_win_size + 1]
            b = (-1) * t_win.reshape(num_of_win_pixels, 1)
            # y_x_list.append((j, i))
            u_v = np.dot(np.linalg.pinv(A), b)
            u_v_list.append(u_v)
            u[i - half_win_size: i + half_win_size + 1, j - half_win_size: j + half_win_size + 1] += u_v[0]
            v[i - half_win_size: i + half_win_size + 1, j - half_win_size: j + half_win_size + 1] += u_v[1]
    # plt.quiver(u, v, color='r')
    # plt.show()
    # warped = warp(im1, y_x_list, u_v_list)
    # plt.imshow(warped, cmap='gray')
    # plt.show()
    # warpCheck(im1, u, v)
    # return np.array(y_x_list).reshape(-1, 2), np.array(u_v_list).reshape(-1, 2)
    return u, v


def opticalFlowPyrLK(img1: np.ndarray, img2: np.ndarray, k: int, stepSize: int, winSize: int) -> np.ndarray:
    """
    :param img1: First image
    :param img2: Second image
    :param k: Pyramid depth
    :param stepSize: The image sample size
    :param winSize: The optical flow window size (odd number)
    :return: A 3d array, with a shape of (m, n, 2),
    where the first channel holds U, and the second V.
    """
    gaus_pyr1 = gaussianPyr(img1, k)
    gaus_pyr2 = gaussianPyr(img2, k)
    plt.imshow(gaus_pyr1[0], cmap='gray')
    plt.show()
    d_u = np.zeros(gaus_pyr1[k - 1].shape)
    d_v = np.zeros(gaus_pyr1[k - 1].shape)
    for i in range(k - 1, -1, -1):
        warped = warpCheck(gaus_pyr1[i], d_u, d_v)
        win_size = int(winSize / (i + 1)) if int(winSize / (i + 1)) % 2 == 1 else int(winSize / (i + 1)) + 1
        dd_u, dd_v = np.array(
            opticalFlowP(warped, gaus_pyr2[i], step_size=stepSize, win_size=int(win_size)))  # int(winSize / k + 8)
        plt.imshow(dd_v, cmap='gray')
        plt.show()
        d_u, d_v = mergeUV(d_u, d_v, 4 * np.floor(dd_u - np.mean(dd_u) + 0.5), 4 * np.floor(dd_v - np.mean(dd_v) + 0.5))
        # d_u, d_v = mergeUV(d_u, d_v, dd_u * 2, dd_v * 2)
        print(d_v.max())
        printRes(warped, gaus_pyr2[i], d_u, d_v, stepSize, int(win_size))
        if i == 0:
            break
        # d_u = gaussExpand(d_u, gaus_pyr1[i - 1].shape) * 2
        # d_v = gaussExpand(d_v, gaus_pyr1[i - 1].shape) * 2
        d_u = blurImage2(np.kron(d_u, np.ones((2, 2), dtype=int)) * 2, 2)
        d_v = blurImage2(np.kron(d_v, np.ones((2, 2), dtype=int)) * 2, 2)
    height, width = img1.shape
    u_v_list, y_x_list = [], []
    for i in range(int(max(stepSize, winSize) / 2), height - int(max(stepSize, winSize) / 2), stepSize):
        for j in range(int(max(stepSize, winSize) / 2), width - int(max(stepSize, winSize) / 2), stepSize):
            y_x_list.append((j, i))
            u_v_list.append((int(d_u[i][j]), int(d_v[i][j])))
    return np.array(y_x_list).reshape(-1, 2), np.array(u_v_list).reshape(-1, 2)


def printRes(warped: np.ndarray, img2: np.ndarray, d_u: np.ndarray, d_v: np.ndarray, stepSize: int, winSize: int):
    height, width = warped.shape
    u_v_list, y_x_list = [], []
    for i in range(int(max(stepSize, winSize) / 2), height - int(max(stepSize, winSize) / 2), stepSize):
        for j in range(int(max(stepSize, winSize) / 2), width - int(max(stepSize, winSize) / 2), stepSize):
            y_x_list.append((j, i))
            u_v_list.append((int(d_u[i][j]), int(d_v[i][j])))
    ptsi, uvi = np.array(y_x_list).reshape(-1, 2), np.array(u_v_list).reshape(-1, 2)
    f, ax = plt.subplots(1, 2)
    ax[0].set_title('warped')
    ax[1].set_title('img2')
    ax[0].imshow(warped, cmap='gray')
    ax[0].quiver(ptsi[:, 0], ptsi[:, 1], uvi[:, 0], uvi[:, 1], color='r')
    ax[1].imshow(img2, cmap='gray')
    ax[1].quiver(ptsi[:, 0], ptsi[:, 1], uvi[:, 0], uvi[:, 1], color='r')
    plt.show()


def mergeUV(u: np.ndarray, v: np.ndarray, d_u: np.ndarray, d_v: np.ndarray):
    # new_u = np.zeros(u.shape)
    # new_v = np.zeros(v.shape)
    # for y in range(u.shape[1]):
    #     for x in range(u.shape[0]):
    #         if y + v[x][y] < u.shape[1] and x + u[x][y] < u.shape[1] < u.shape[0]:
    #             new_u[x + u[x][y]][y + v[x][y]] = u[x][y]
    #             new_v[x + u[x][y]][y + v[x][y]] = v[x][y]
    # new_u += d_u
    # new_v += d_v
    for y in range(u.shape[1]):
        for x in range(u.shape[0]):
            if y + v[x][y] < u.shape[1] and x + u[x][y] < u.shape[0]:
                move_u, move_v = int(u[x][y]), int(v[x][y])
                u[x][y] += d_u[x + move_u][y + move_v]
                d_u[x + move_u][y + move_v] = 0
                v[x][y] += d_v[x + move_u][y + move_v]
                d_v[x + move_u][y + move_v] = 0
    return u, v


# ---------------------------------------------------------------------------
# ------------------------ Image Alignment & Warping ------------------------
# ---------------------------------------------------------------------------


def findTranslationLK(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: image 1 in grayscale format.
    :param im2: image 1 after Translation.
    :return: Translation matrix by LK.
    """
    img1 = im1
    img2 = im2
    k = 6
    gaus_pyr1 = gaussianPyr(img1, k)
    gaus_pyr2 = gaussianPyr(img2, k)
    u, v = 0, 0
    warping_mat = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    for i in range(k - 1, -1, -1):
        print(warping_mat)
        plt.imshow(gaus_pyr1[i], cmap='gray')
        plt.show()
        gaus_pyr1[i] = warpImages(gaus_pyr1[i], np.zeros(gaus_pyr1[i].shape), warping_mat)
        plt.imshow(gaus_pyr1[i], cmap='gray')
        plt.show()
        img_size = min(gaus_pyr1[i].shape[0], gaus_pyr1[i].shape[1]) - 1
        y_x, u_v = np.array(opticalFlow(gaus_pyr1[i], gaus_pyr2[i], step_size=img_size, win_size=img_size))
        if len(u_v) != 0:
            u = 2 * u_v[0][0]
            v = 2 * u_v[0][1]
            warping_mat = np.array([[1, 0, u], [0, 1, v], [0, 0, 1]])
        else:
            u, v = 2 * u, 2 * v
            warping_mat = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    print("u = " + str(u) + ", v = " + str(v))
    pass


def findRigidLK(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Rigid.
    :return: Rigid matrix by LK.
    """
    pass


# https://stackoverflow.com/questions/58174390/how-to-detect-image-translation-with-only-numpy-and-pil
def findTranslationCorr(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Translation.
    :return: Translation matrix by correlation.
    """
    y, x, max_point_value = maxCorrelationPoint(im1, im2)
    y_distance = im1.shape[0] // 2 - y
    x_distance = im1.shape[1] // 2 - x
    warping_mat = np.array([[1, 0, x_distance], [0, 1, y_distance], [0, 0, 1]])
    return warping_mat


# https://stackoverflow.com/questions/23619269/calculating-translation-value-and-rotation-angle-of-a-rotated-2d-image
def findRigidCorr(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Rigid.
    :return: Rigid matrix by correlation.
    """
    res_y, res_x, res_max = -1, -1, -1
    res_theta = -1
    for i in range(0, int(2.0 * np.pi * 10), 5):
        theta = float(i) / 10.0
        rotat_mat = np.array([[np.cos(theta), -np.sin(theta), 0],
                              [np.sin(theta), np.cos(theta), 0],
                              [0, 0, 1]])
        rotated_im1 = warpImages(im1, np.zeros(im1.shape), rotat_mat)
        y, x, max_point_value = maxCorrelationPoint(rotated_im1, im2)
        if max_point_value > res_max:
            res_y, res_x, res_max = y, x, max_point_value
            res_theta = theta
    y_distance = im1.shape[0] // 2 - res_y
    x_distance = im1.shape[1] // 2 - res_x
    warping_mat = np.array([[np.cos(res_theta), -np.sin(res_theta), x_distance],
                            [np.sin(res_theta), np.cos(res_theta), y_distance],
                            [0, 0, 1]])
    return warping_mat


# TODO: Inverse?
# https://github.com/ZhihaoZhu/Image-Warping
def warpImages(im1: np.ndarray, im2: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: input image 2 in grayscale format.
    :param T: is a 3x3 matrix such that each pixel in image 2
    is mapped under homogenous coordinates to image 1 (p2=Tp1).
    :return: warp image 2 according to T and display both image1
    and the wrapped version of the image2 in the same figure.
    """
    for i in range(0, im2.shape[0]):
        for j in range(0, im2.shape[1]):
            # new_coordinates = np.linalg.inv(T).dot(np.array([i, j, 1]))
            new_coordinates = np.linalg.inv(T).dot(np.array([j, i, 1]))
            new_j, new_i = int(new_coordinates[0]), int(new_coordinates[1])
            # print("i:"+str(i)+",j:"+str(j)+" new_i:"+str(new_i)+",new_j:"+str(new_j))
            if 0 <= new_i < im1.shape[0] and 0 <= new_j < im1.shape[1]:
                im2[i, j] = im1[new_i, new_j]
    return im2.astype(im1.dtype)


def maxCorrelationPoint(im1: np.ndarray, im2: np.ndarray):
    # get rid of the averages, otherwise the results are not good
    im1_gray = im1 - np.mean(im1)
    im2_gray = im2 - np.mean(im2)
    # calculate the correlation image (without scipy)
    pad = np.max(im1_gray.shape) // 2
    fft1 = np.fft.fft2(np.pad(im1_gray, pad))
    fft2 = np.fft.fft2(np.pad(im2_gray, pad))
    prod = fft1 * fft2.conj()
    result_full = np.fft.fftshift(np.fft.ifft2(prod))
    corr = result_full.real[1 + pad:-pad + 1, 1 + pad:-pad + 1]
    max_point_value = np.argmax(corr)
    y, x = np.unravel_index(max_point_value, corr.shape)
    return y, x, np.max(corr)


def warpCheck(image: np.ndarray, u: np.ndarray, v: np.ndarray):
    result = np.zeros(image.shape) - 1
    print(str(result.shape) + str("gggg"))
    for y in range(image.shape[1] - 2):
        for x in range(image.shape[0] - 2):
            # if int(x + u[x][y]) < result.shape[0] and int(y + v[x][y]) < result.shape[1]\
            #         and x < image.shape[0] and y < image.shape[1]:
            if x < u.shape[0] and y < v.shape[1] and int(x + u[x][y]) < result.shape[0] and int(y + v[x][y]) < \
                    result.shape[1]:
                result[int(x + u[x][y])][int(y + v[x][y])] = image[x][y]
    mask = np.array([[1 if result[j][i] == -1 else 0 for i in range(image.shape[1])] for j in range(image.shape[0])])
    # dst = cv2.inpaint(result / 255.0, mask / 1.0, 3, cv2.INPAINT_TELEA) # INPAINT_NS
    intepolated = cv2.inpaint(cv2.convertScaleAbs(result), cv2.convertScaleAbs(mask), 1, cv2.INPAINT_TELEA)
    plt.imshow(result, cmap='gray')
    plt.show()
    return intepolated


# ---------------------------------------------------------------------------
# --------------------- Gaussian and Laplacian Pyramids ---------------------
# ---------------------------------------------------------------------------


def gaussianPyr(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """
    Creates a Gaussian Pyramid
    :param img: Original image
    :param levels: Pyramid depth
    :return: Gaussian pyramid (list of images)
    """
    pyr = []
    height, width = img.shape[0], img.shape[1]
    for level in range(levels):
        pyr.append(img)
        img = blurImage2(img, 5)
        img = np.array([[img[j][i] for i in range(0, width, 2)] for j in range(0, height, 2)])
        height, width = int(height / 2), int(width / 2)
    return pyr


def laplaceianReduce(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """
    Creates a Laplacian pyramid
    :param img: Original image
    :param levels: Pyramid depth
    :return: Laplacian Pyramid (list of images)
    """
    pyr = []
    gaus_pyr = gaussianPyr(img, levels)
    for level in range(levels - 1):
        gaus_curr_level = gaus_pyr[level]
        gaus_prev_level = gaus_pyr[level + 1]
        gaus_expended_prev_level = gaussExpand(gaus_prev_level, gaus_curr_level.shape)
        pyr.append(gaus_curr_level - gaus_expended_prev_level)
    pyr.append(gaus_pyr[-1])
    return pyr


def gaussExpand(img: np.ndarray, target_shape: np.shape):
    height, width = img.shape[0] * 2, img.shape[1] * 2
    gaus_expended_prev_level = np.zeros(target_shape)
    for i in range(0, width, 2):
        for j in range(0, height, 2):
            gaus_expended_prev_level[j, i] = img[j // 2][i // 2]
    gaus_expended_prev_level = np.clip(blurImage2(gaus_expended_prev_level, 5) * 4, 0, 1)
    return gaus_expended_prev_level


def laplaceianExpand(lap_pyr: List[np.ndarray]) -> np.ndarray:
    """
    Restores the original image from a laplacian pyramid
    :param lap_pyr: Laplacian Pyramid
    :return: Original image
    """
    image = lap_pyr[-1]
    for i in range(len(lap_pyr) - 2, -1, -1):
        image = np.clip(lap_pyr[i] + gaussExpand(image, lap_pyr[i].shape), 0, 1)
    return image


def pyrBlend(img_1: np.ndarray, img_2: np.ndarray,
             mask: np.ndarray, levels: int) -> (np.ndarray, np.ndarray):
    """
    Blends two images using PyramidBlend method
    :param img_1: Image 1
    :param img_2: Image 2
    :param mask: Blend mask
    :param levels: Pyramid depth
    :return: (Naive blend, Blended Image)
    """
    lap_pyr1 = laplaceianReduce(img_1, levels)
    lap_pyr2 = laplaceianReduce(img_2, levels)
    gauss_pyr = gaussianPyr(mask, levels)
    lap_pyr3 = [np.array(gauss_pyr[i] * lap_pyr1[i] + (1 - gauss_pyr[i]) * lap_pyr2[i]) for i in range(levels)]
    naive = np.array([[(img_2[j][i] if (mask[j][i].any() < 0.5) else img_1[j][i])
                       for i in range(mask.shape[1])] for j in range(mask.shape[0])])
    return naive, laplaceianExpand(lap_pyr3)


def blurImage2(in_image: np.ndarray, k_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel using OpenCV built-in functions
    :param in_image: Input image
    :param k_size: Kernel size
    :return: The Blurred image
    """
    gaussian_kernel = cv2.getGaussianKernel(k_size, 0)
    gaussian_kernel = gaussian_kernel * gaussian_kernel.T
    return cv2.filter2D(in_image, -1, gaussian_kernel, borderType=cv2.BORDER_REPLICATE)
