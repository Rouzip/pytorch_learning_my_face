import os
import random

import dlib
import cv2

# the dir to save my picture
out_pic_dir = './image/my_face'
if not os.path.exists(out_pic_dir):
    os.makedirs(out_pic_dir)

# the picture size
size = 64


def adjust_pic(img, light=1, bias=0):
    '''
    adjust the picture light and the contrast
    '''
    h = img.shape[0]
    w = img.shape[1]
    for i in range(h):
        for j in range(w):
            for k in range(3):
                tmp = int(img[i, j, k] * light + bias)
                if tmp > 255:
                    tmp = 255
                elif tmp < 0:
                    tmp = 0
                img[i, j, k] = tmp
    return img

# # get the front face detector by dlib
detector = dlib.get_frontal_face_detector()
# get the camera in the computer
camera = cv2.VideoCapture(0)

for num in range(10000):
    _, img = camera.read()
    # convert the img to gray
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # get the fectures in the img
    dets = detector(img, 1)

    for i in dets:
        x1 = i.top() if i.top() > 0 else 0
        y1 = i.bottom() if i. bottom() > 0 else 0
        x2 = i.left() if i.left() > 0 else 0
        y2 = i.right() if i.right() > 0 else 0

    # get the img face exactly and adjust the img
    face = img[x1:y1, x2:y2]
    face = adjust_pic(face,
                      random.uniform(0.5, 1.5),
                      random.randint(-50, 50))

    face = cv2.resize(face, (size, size))
    if not len(face):
        continue
    cv2.imwrite(out_pic_dir+'/'+str(num)+'.jpg', face)

else:
    print('finish get the picture')
