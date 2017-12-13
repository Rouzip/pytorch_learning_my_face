import os
import shutil
import random

# get the train image set
image_dir = os.getcwd()+'/image'
test_dir = os.getcwd()+'/test_image'


for root, dirs, files in os.walk(image_dir):
    image_size = len(files)
    if image_size == 0:
        continue
    else:
        # get 20% in images to test
        test_size = image_size // 10 * 2
        file_names = random.sample(files, test_size)
        destination = root.split('/')[-1]
        for image in file_names:
            shutil.move(root+'/'+image, test_dir+'/'+destination+'/'+image)
