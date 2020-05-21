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

VGG_types = { #M stands for MaxPooling operation
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG_net(nn.Module): #Inheriting from nn. Module provides functionality to your component. For example, it makes it keep track of its trainable parameters, you can swap it between CPU and GPU.
    def __init__(self, in_channels=3, num_classes=1000): #VGG was trained for 1000 objects detection
        super(VGG_net, self).__init__() # Calling the superclass of nn.Module
        self.in_channels = in_channels
        self.conv_layers = self.create_conv_layers(VGG_types['VGG19']) #If we want to use any other VGG architecture, we need to change the name in VGG_types

        self.fcs = nn.Sequential( # Adds those layers to our models in a sequence
            nn.Linear(512 * 8 * 8, 4096), #Fully connected layer, takes the flattened CNN layer as input and gives an output of 4096 values
            nn.ReLU(), #Rectified Linear Unit
            nn.Dropout(p=0.5), #Dropout ignores values, surprisingly works well, we've chose probability of dropping out as 0.5
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x): # The other in which  the input will get passed through the different layers
        x = self.conv_layers(x) #Defined below, input passed through all the Convolutional layers
        x = x.reshape(x.shape[0], -1) #Before passing into fully connected layers, we flatten or make it unidimensional.
        x = self.fcs(x) #Now the input is passed through all linear layers, fcs is definted above
        return x #Final output is returned

    def create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == int:
                out_channels = x
        #Adding convolutional layers, as per our requirement
                layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                     kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                           nn.BatchNorm2d(x),
                           nn.ReLU()]
                in_channels = x
            elif x == 'M': #If it encounters 'M', adds a Max-pooling layer
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]

        return nn.Sequential(*layers)

#We load the weights of the network we have trained and keep it in eval() to make sure no weights get updated

model = VGG_net(3, 2)
model.load_state_dict(torch.load("C:/Users/Saurabh/Desktop/Mask Detector/weights2.pt"))
model.eval()

#We define a function which detects the mask realtime using openCV

def maskDetection(im_pil,img):
    im_pil = tt.ToTensor()(im_pil).unsqueeze_(0)
    listt = F.softmax(model(im_pil),dim=1)

    if listt[0][0] > 0.8: #If probability of 0th index is more than 0.8, then print masked
        cv2.putText(img, "Masked", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    elif listt[0][1] > 0.8: #If probability of 1st index is more than 0.8, then print Unmasked
        cv2.putText(img, "Unmasked", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    else: #Else print that it's not clear
       cv2.putText(img, "Please stay still or fix your lighting.", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

#Press q to exit

fc= cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
a=0
while 1:

    a=a+1
    if(a%10==0):
      frame=cap.read()
      a=1
    ret, img = cap.read() #Capturing video feed in form of image
    cv2.imwrite('C:/Users/Saurabh/Desktop/Mask Detector/TestFaces/test.jpg',img)
    im_pill = Image.open('C:/Users/Saurabh/Desktop/Mask Detector/TestFaces/test.jpg')
    faces = fc.detectMultiScale(img, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2) #Draws a rectangle around the face

    im_pill = im_pill.resize((256,256))
    maskDetection(im_pill,img)
    cv2.imshow('img', img)

    if (cv2.waitKey(1) & 0xFF == ord('q')):
        break
cap.release()
cv2.destroyAllWindows()
