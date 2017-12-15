import random
import os

import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
from torch.nn.functional import relu
from torch.utils.data import DataLoader
from torchvision.transforms import Grayscale


size = 64


class Train(nn.Module):
    '''
    the model to recognize my face
    '''

    def __init__(self):
        super(Train, self).__init__()
        # input channel 1,output channel 6,kernel 3*3,default stride 1,default
        # padding 0
        self.conv1 = nn.Conv2d(1, 6, kernel_size=3)
        # input channel 6,output channel 16,kernel 3*3,default stride 1,default
        # padding 0
        self.conv2 = nn.Conv2d(6, 16, kernel_size=3)
        # kernel size 2,default stride kernel
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16*14*14, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)
        self.softmax = nn.Softmax(dim=None)

    def forward(self, x):
        # get 31*31
        x = self.pool(F.relu(self.conv1(x)))
        # get 14*14
        x = self.pool(F.relu(self.conv2(x)))
        # this model is ensure the size of the picture 64*64
        x = x.view(-1, 14*14*16)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.softmax(x)
        return x


# create the model,build the optimizer and the loss
model = Train()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
criterion = nn.CrossEntropyLoss()

# load the pictures
transform = transforms.Compose(
    [transforms.Grayscale(1),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

data = datasets.ImageFolder(root=os.getcwd()+'/image', transform=transform)
data_loader = DataLoader(dataset=data, batch_size=4,
                         num_workers=8, shuffle=True)


# 循环迭代十次
for epoch in range(10):

    running_loss = 0.0
    for i, data in enumerate(data_loader):
        # get the inputs
        inputs, labels = data
        # wrap them in Varialbe
        inputs, labels = Variable(inputs), Variable(labels)

        # zero the parameter gradients
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        # back
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.data[0]
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0


torch.save(model, 'model.pkl')
print('train finished!')
