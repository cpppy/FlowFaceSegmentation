import cv2
import numpy as np
import os

if __name__=='__main__':
    img_dir = '/Users/atom/data/FlowFaceSegmentation/data/sample'
    img_fn_list = os.listdir(img_dir)
    print(img_fn_list)
    img_1 = cv2.imread(os.path.join(img_dir, '0a5a3db4535c8b7c21610437b1ca37d7_04.jpg'))
    img_2 = cv2.imread(os.path.join(img_dir, '0a5a3db4535c8b7c21610437b1ca37d7_05.jpg'))
    img_1 = cv2.resize(img_1, dsize=(0, 0), fx=0.3, fy=0.3)
    img_2 = cv2.resize(img_2, dsize=(0, 0), fx=0.3, fy=0.3)


    prvs = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(img_1)
    dis = cv2.DISOpticalFlow_create(2)

    next = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
    flow = dis.calc(prvs, next, None, )
    # flow = cv.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    hsv[..., 2] = 255
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    result = cv2.hconcat([img_1, img_2, bgr])
    cv2.imshow('result', result)
    cv2.waitKey(0)

