import os

import cv2
import dlib

input_dir = './lfw'
output_dir = os.getcwd() + '/image/other_face'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# use the dlib face front detector
detector = dlib.get_frontal_face_detector()

# adjust the picture size into 64*64
size = 64
index = 0
# change the other people pictures into feature pictures
for path, _, files in os.walk(input_dir):
    for file in files:
        if file.endswith('.jpg'):
            img_path = path + '/' + file
            # load the gray picture
            gray_img = cv2.imread(img_path, 0)

            dets = detector(gray_img, 1)
            for i in dets:
                x1 = i.top() if i.top() > 0 else 0
                y1 = i.bottom() if i. bottom() > 0 else 0
                x2 = i.left() if i.left() > 0 else 0
                y2 = i.right() if i.right() > 0 else 0

                # adjust the origin picture to train network
                face = gray_img[x1:y1, x2:y2]
                face = cv2.resize(face, (size, size))
                cv2.imwrite(output_dir+'/'+str(index)+'.jpg', face)
                index += 1
