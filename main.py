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

    u = np.array([[1, 0, 0],
                  [0, 0, 0]])
    v = np.array([[1, 0, 0],
                  [0, 0, 0]])
    d_u = np.array([[0, 0, 0],
                    [0, 0, 0]])
    d_v = np.array([[0, 0, 0],
                    [0, 1, 0]])

    new_u, new_v = mergeUV(u, v, d_u, d_v)

    print(new_v)


