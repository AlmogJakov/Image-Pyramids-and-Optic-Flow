from ex3_utils import *
import time


# ---------------------------------------------------------------------------
# ------------------------ Lucas Kanade optical flow ------------------------
# ---------------------------------------------------------------------------


def lkDemo(img_path):
    print("LK Demo")

    img_1 = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    img_1 = cv2.resize(img_1, (0, 0), fx=.5, fy=0.5)
    t = np.array([[1, 0, -.2],
                  [0, 1, -.1],
                  [0, 0, 1]], dtype=float)
    img_2 = cv2.warpPerspective(img_1, t, img_1.shape[::-1])
    st = time.time()
    pts, uv = opticalFlow(img_1.astype(float), img_2.astype(float), step_size=20, win_size=5)
    et = time.time()

    print("Time: {:.4f}".format(et - st))
    print(np.median(uv,0))
    print(np.mean(uv,0))

    displayOpticalFlow(img_2, pts, uv)


def hierarchicalkDemo(img_path):
    """
    ADD TEST
    :param img_path: Image input
    :return:
    """
    print("Hierarchical LK Demo")
    print("LK Demo")
    # img_path1 = 'input/sphere1.jpg'
    # img_path2 = 'input/sphere2.jpg'
    # img_1 = cv2.cvtColor(cv2.imread(img_path1), cv2.COLOR_BGR2GRAY)
    # img_2 = cv2.cvtColor(cv2.imread(img_path2), cv2.COLOR_BGR2GRAY)
    # st = time.time()
    # STEP_SIZE, WIN_SIZE = 10, 29
    # pts, uv = opticalFlow(img_1.astype(float), img_2.astype(float), step_size=STEP_SIZE, win_size=WIN_SIZE)
    # ptsi, uvi = iterativeopticalFlow(img_1.astype(float), img_2.astype(float), step_size=STEP_SIZE, win_size=WIN_SIZE)
    # et = time.time()
    #
    # print("Time: {:.4f}".format(et - st))
    # print(np.median(uv, 0))
    # print(np.mean(uv, 0))
    #
    # f, ax = plt.subplots(1, 2)
    # ax[0].set_title('Optical Flow')
    # ax[1].set_title('Iterative Optical Flow')
    # ax[0].imshow(img_2, cmap='gray')
    # ax[0].quiver(pts[:, 0], pts[:, 1], uv[:, 0], uv[:, 1], color='r')
    # ax[1].imshow(img_2, cmap='gray')
    # ax[1].quiver(ptsi[:, 0], ptsi[:, 1], uvi[:, 0], uvi[:, 1], color='r')
    # plt.show()
    img_path1 = 'input/Dense_Motion_A.jpg'
    img_path2 = 'input/Dense_Motion_B.jpg'
    img_1 = cv2.cvtColor(cv2.imread(img_path1), cv2.COLOR_BGR2GRAY)
    img_2 = cv2.cvtColor(cv2.imread(img_path2), cv2.COLOR_BGR2GRAY)
    STEP_SIZE, WIN_SIZE = 5, 29
    opticalFlowPyrLK(img_1.astype(float), img_2.astype(float), 5, stepSize=STEP_SIZE, winSize=WIN_SIZE)
    pass


def compareLK(img_path):
    """
    ADD TEST
    Compare the two results from both functions.
    :param img_path: Image input
    :return:
    """
    print("Compare LK & Hierarchical LK")

    pass


def displayOpticalFlow(img: np.ndarray, pts: np.ndarray, uvs: np.ndarray):
    plt.imshow(img, cmap='gray')
    plt.quiver(pts[:, 0], pts[:, 1], uvs[:, 0], uvs[:, 1], color='r')

    plt.show()


# ---------------------------------------------------------------------------
# ------------------------ Image Alignment & Warping ------------------------
# ---------------------------------------------------------------------------


def imageWarpingDemo(img_path):
    """
    ADD TEST
    :param img_path: Image input
    :return:
    """
    print("Image Warping Demo")
    img_path1 = 'input/sphere1.jpg'
    T = [[1,2,0],[2,1,0],[0,2,1]]
    img1 = np.array(cv2.imread(img_path1))
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = np.array(np.zeros(img1.shape))
    res = warpImages(img1, img2, T)
    f, ax = plt.subplots(1, 2)
    ax[0].set_title('Original Image')
    ax[1].set_title('Warped Image')
    ax[0].imshow(img1)
    ax[1].imshow(res)
    plt.show()
    pass


# ---------------------------------------------------------------------------
# --------------------- Gaussian and Laplacian Pyramids ---------------------
# ---------------------------------------------------------------------------


def pyrGaussianDemo(img_path):
    print("Gaussian Pyramid Demo")

    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) / 255
    #img = cv2.cvtColor(np.array(img).astype('float32') * 255, cv2.COLOR_RGB2GRAY) / 255
    lvls = 4
    gau_pyr = gaussianPyr(img, lvls)
    h, w = gau_pyr[0].shape[:2]
    canv_h = h
    widths = np.cumsum([w // (2 ** i) for i in range(lvls)])
    widths = np.hstack([0, widths])
    canv_w = widths[-1]
    canvas = np.zeros((canv_h, canv_w, 3))

    for lv_idx in range(lvls):
        h = gau_pyr[lv_idx].shape[0]
        canvas[:h, widths[lv_idx]:widths[lv_idx + 1], :] = gau_pyr[lv_idx]

    plt.imshow(canvas)
    plt.show()


def pyrLaplacianDemo(img_path):
    print("Laplacian Pyramid Demo")

    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY) / 255
    # img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) / 255
    lvls = 7

    lap_pyr = laplaceianReduce(img, lvls)
    re_lap = laplaceianExpand(lap_pyr)

    f, ax = plt.subplots(2, lvls + 1)
    plt.gray()
    for i in range(lvls):
        ax[0, i].imshow(lap_pyr[i])
        ax[1, i].hist(lap_pyr[i].ravel(), 256, [lap_pyr[i].min(), lap_pyr[i].max()])

    ax[0, -1].set_title('Original Image')
    ax[0, -1].imshow(re_lap)
    ax[1, -1].hist(re_lap.ravel(), 256, [0, 1])
    plt.show()


def blendDemo():
    im1 = cv2.cvtColor(cv2.imread('input/sunset.jpg'), cv2.COLOR_BGR2RGB) / 255
    im2 = cv2.cvtColor(cv2.imread('input/cat.jpg'), cv2.COLOR_BGR2RGB) / 255
    mask = cv2.cvtColor(cv2.imread('input/mask_cat.jpg'), cv2.COLOR_BGR2RGB) / 255

    n_blend, im_blend = pyrBlend(im1, im2, mask, 4)

    f, ax = plt.subplots(2, 3)
    ax[0, 0].imshow(im1)
    ax[0, 1].imshow(im2)
    ax[0, 2].imshow(mask)
    ax[1, 0].imshow(n_blend)
    ax[1, 1].imshow(np.abs(n_blend - im_blend))
    ax[1, 2].imshow(im_blend)

    plt.show()

    cv2.imwrite('sunset_cat.png', cv2.cvtColor((im_blend * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))


def main():
    print("ID:", myID())

    img_path = 'input/boxMan.jpg'
    #lkDemo(img_path)
    #hierarchicalkDemo(img_path)
    #compareLK(img_path)
    #
    # imageWarpingDemo(img_path)
    #
    #pyrGaussianDemo('input/pyr_bit.jpg')
    #pyrLaplacianDemo('input/pyr_bit.jpg')
    #blendDemo()


if __name__ == '__main__':
    main()
