import cv2
import dlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms


class Train(nn.Module):
    '''
    the model to recognize my face
    '''

    def __init__(self):
        super(Train, self).__init__()
        # input channel 1,output channel 6,kernel 3*3,default stride 1,default
        # padding 0
        self.conv1 = nn.Conv2d(1, 6, kernel_size=(3, 3))
        # input channel 6,output channel 16,kernel 3*3,default stride 1,default
        # padding 0
        self.conv2 = nn.Conv2d(6, 16, kernel_size=(3, 3))
        # kernel size 2,default stride kernel
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16*14*14, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        # get 31*31
        x = self.pool(F.relu(self.conv1(x)))  # 6*31*31
        # get 14*14
        x = self.pool(F.relu(self.conv2(x)))  # 16*14*14
        # this model is ensure the size of the picture 64*64
        x = x.view(-1, 14*14*16)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = torch.load('model.pkl').eval()
# set the model verification mode


# open the camera
camera = cv2.VideoCapture(0)
# get the img
success, img = camera.read()
detecter = dlib.get_frontal_face_detector()
dets = detecter(img, 1)
for i in dets:
    for i in dets:
        x1 = i.top() if i.top() > 0 else 0
        y1 = i.bottom() if i. bottom() > 0 else 0
        x2 = i.left() if i.left() > 0 else 0
        y2 = i.right() if i.right() > 0 else 0

    # get the img face exactly and adjust the img
    face = img[x1:y1, x2:y2]
    face = cv2.resize(face, (64, 64))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face = torch.from_numpy(face)
    face = face.float()
    face = face.unsqueeze(0)
    face = face.unsqueeze(0)
    face = Variable(face)

    print(face)

    out = model(face)
    print(123)
    print(out)
