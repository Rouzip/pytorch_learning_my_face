import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader


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
        self.drop = nn.Dropout()

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

# load the pictures
transform = transforms.Compose(
    [transforms.Grayscale(1),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

test_data = datasets.ImageFolder(root=os.getcwd()+'/test_image',
                                 transform=transform)
test_loader = DataLoader(dataset=test_data, batch_size=4,
                         num_workers=8, shuffle=True)


model = torch.load('model.pkl').eval()

correct = 0.0
total = 0.0
for data in test_loader:
    images, labels = data
    outputs = model(Variable(images))
    _, predict = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predict == labels).sum()
print(correct/total)
