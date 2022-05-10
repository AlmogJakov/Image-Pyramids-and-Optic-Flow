# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import math

import numpy as np
from cv2 import cv2
import matplotlib.pyplot as plt

from ex3_utils import *


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    print("Your OpenCV version is: " + cv2.__version__)

    img_path1 = 'input/Dense_Motion_A.jpg'
    img_path2 = 'input/Dense_Motion_B.jpg'
    img_1 = cv2.cvtColor(cv2.imread(img_path1), cv2.COLOR_BGR2GRAY)

    ############################ findTranslationCorr TEST ############################
    # trans_mat = np.array([[1, 0, 10],
    #                       [0, 1, 50],
    #                       [0, 0, 1]])
    # trans_img = warpImages(img_1, np.zeros(img_1.shape), trans_mat)
    # plt.imshow(trans_img, cmap='gray')
    # plt.show()
    # print(findTranslationCorr(img_1, trans_img))

    ############################ findRigidCorr TEST ############################
    theta = 1.0
    rotat_mat = np.array([[np.cos(theta), -np.sin(theta), 0],
                          [np.sin(theta),  np.cos(theta), 0],
                          [0,              0,             1]])
    rotat_img = warpImages(img_1, np.zeros((img_1.shape[0], img_1.shape[1])), rotat_mat)
    plt.imshow(rotat_img, cmap='gray')
    plt.show()
    print(findTranslationCorr(img_1, rotat_img))

    img = cv2.imread(img_path1, 0)
    rows, cols = img.shape
    M = np.float32([[1, 0, 10], [0, 1, 50]])
    dst = cv2.warpAffine(img, M, (cols,rows))
    plt.imshow(dst, cmap='gray')
    plt.show()
    #res = findTranslationLK(img_1, img_2)
    #findRigidCorr(img_1, rotat_img)
    #print(findTranslationCorr(img_1, trans_img))
