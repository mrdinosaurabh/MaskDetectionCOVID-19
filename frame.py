#Importing all the necessary packages

import os
import torch
import torchvision
import torch.nn as nn
from torchvision import transforms as tt
import PIL
from PIL import Image
import torch.nn.functional as F
import cv2
import numpy as np

#Thanks to Aladdin Persson (YouTube) that I could get this code for VGG architecture.

VGG_types = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG_net(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000):
        super(VGG_net, self).__init__()
        self.in_channels = in_channels
        self.conv_layers = self.create_conv_layers(VGG_types['VGG19'])

        self.fcs = nn.Sequential(
            nn.Linear(512 * 8 * 8, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fcs(x)
        return x

    def create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == int:
                out_channels = x

                layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                     kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                           nn.BatchNorm2d(x),
                           nn.ReLU()]
                in_channels = x
            elif x == 'M':
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]

        return nn.Sequential(*layers)

#We load the weights of the network we have trained

model = VGG_net(3, 2)
model.load_state_dict(torch.load("C:/Users/Saurabh/Desktop/Mask Detector/weights2.pt"))
model.eval()

#We define a function which detects the mask realtime using openCV

def maskDetection(im_pil,img):
    im_pil = tt.ToTensor()(im_pil).unsqueeze_(0)
    listt = F.softmax(model(im_pil),dim=1)

    if listt[0][0] > 0.8:
        cv2.putText(img, "Masked", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    elif listt[0][1] > 0.8:
        cv2.putText(img, "Unmasked", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    else:
       cv2.putText(img, "Please stay still or fix your lighting.", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

#Press q to exit

fc= cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
while 1:
    ret, img = cap.read()
    cv2.imwrite('C:/Users/Saurabh/Desktop/Mask Detector/TestFaces/test.jpg',img)
    im_pill = Image.open('C:/Users/Saurabh/Desktop/Mask Detector/TestFaces/test.jpg')
    faces = fc.detectMultiScale(img, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    im_pill = im_pill.resize((256,256))
    maskDetection(im_pill,img)
    cv2.imshow('img', img)

    if (cv2.waitKey(1) & 0xFF == ord('q')):
        break
cap.release()
cv2.destroyAllWindows()
