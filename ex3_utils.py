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
    height, width = img2.shape
    half_win_size, num_of_win_pixels = win_size // 2, win_size ** 2
    u_v_list, y_x_list = [], []
    for i in range(step_size, height, step_size):
        for j in range(step_size, width, step_size):
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
MAX_LAMDA_RATIO = 25


def __acceptable_eigenvalues(A: np.ndarray) -> bool:
    squared_matrix = np.dot(np.transpose(A), A)
    lamda1, lamda2 = np.sort(np.linalg.eigvals(squared_matrix))
    if lamda1 > MIN_LAMDA and lamda2 > MIN_LAMDA and (lamda2 / lamda1) < MAX_LAMDA_RATIO:
        return True
    return False


def __get_directions(img: np.ndarray):
    v = np.array([[1, 0, -1]])
    X = cv2.filter2D(img, -1, v)
    Y = cv2.filter2D(img, -1, v.T)
    return X, Y


def opticalFlowPyrLK(img1: np.ndarray, img2: np.ndarray, k: int,
                     stepSize: int, winSize: int) -> np.ndarray:
    """
    :param img1: First image
    :param img2: Second image
    :param k: Pyramid depth
    :param stepSize: The image sample size
    :param winSize: The optical flow window size (odd number)
    :return: A 3d array, with a shape of (m, n, 2),
    where the first channel holds U, and the second V.
    """
    pass


def iterativeopticalFlow(im1: np.ndarray, im2: np.ndarray, step_size=10,
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
    # if we change I_x, I_y to img1 we need to change t_win to subtract img2, img1
    height, width = img2.shape
    half_win_size, num_of_win_pixels = win_size // 2, win_size ** 2
    u_v_list, y_x_list = [], []
    for i in range(step_size, height, step_size):
        for j in range(step_size, width, step_size):
            x_win = I_x[i - half_win_size: i + half_win_size + 1, j - half_win_size: j + half_win_size + 1]
            y_win = I_y[i - half_win_size: i + half_win_size + 1, j - half_win_size: j + half_win_size + 1]
            A = np.hstack((x_win.reshape(num_of_win_pixels, 1), y_win.reshape(num_of_win_pixels, 1)))
            if not __acceptable_eigenvalues(A):
                continue
            u, v = 0, 0
            t_win = np.subtract(img1[i - half_win_size: i + half_win_size + 1, j - half_win_size: j + half_win_size + 1]
                , img2[i + u - half_win_size: i + u + half_win_size + 1,
                  j + v - half_win_size: j + v + half_win_size + 1])
            b = (-1) * t_win.reshape(num_of_win_pixels, 1)
            while True:
                [du], [dv] = np.array(np.dot(np.linalg.pinv(A), b)).astype('int')
                if du != 0 or dv != 0:
                    u += du
                    v += dv
                if i + u <= half_win_size or j + v <= half_win_size or i + u >= height - half_win_size or j + v >= width - half_win_size:
                    break
                t_win_temp = np.subtract(img1[i - half_win_size: i + half_win_size + 1, j - half_win_size: j + half_win_size + 1]
                    , img2[i + u - half_win_size: i + u + half_win_size + 1, j + v - half_win_size: j + v + half_win_size + 1])
                if t_win_temp.sum() < t_win.sum():
                    t_win = t_win_temp
                else:
                    break
                b = (-1) * t_win.reshape(num_of_win_pixels, 1)
            y_x_list.append((j, i))
            u_v = np.dot(np.linalg.pinv(A), b)
            u_v_list.append(u_v)
    return np.array(y_x_list).reshape(-1, 2), np.array(u_v_list).reshape(-1, 2)

# ---------------------------------------------------------------------------
# ------------------------ Image Alignment & Warping ------------------------
# ---------------------------------------------------------------------------


def findTranslationLK(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: image 1 in grayscale format.
    :param im2: image 1 after Translation.
    :return: Translation matrix by LK.
    """
    pass


def findRigidLK(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Rigid.
    :return: Rigid matrix by LK.
    """
    pass


def findTranslationCorr(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Translation.
    :return: Translation matrix by correlation.
    """
    pass


def findRigidCorr(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Rigid.
    :return: Rigid matrix by correlation.
    """
    pass


def warpImages(im1: np.ndarray, im2: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: input image 2 in grayscale format.
    :param T: is a 3x3 matrix such that each pixel in image 2
    is mapped under homogenous coordinates to image 1 (p2=Tp1).
    :return: warp image 2 according to T and display both image1
    and the wrapped version of the image2 in the same figure.
    """
    pass


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
