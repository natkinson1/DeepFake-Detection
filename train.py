import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import numpy as np
import torch
from torchvision import models, transforms, datasets
import torch.nn.functional as F
from torch import nn, optim
import face_detector
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_1 = models.vgg19(pretrained=True)
model_2 = models.vgg19(pretrained=True)
model_3 = models.vgg19(pretrained=True)
model_4 = models.vgg19(pretrained=True)
model_5 = models.vgg19(pretrained=True)

model_list = [model_1, model_2, model_3, model_4, model_5]

class Network1(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.layer1 = nn.Linear(25088,5000).cuda()
        self.layer2 = nn.Linear(5000, 500).cuda()
        self.layer3 = nn.Linear(500, 2).cuda()

    def forward(self, x):
        
        #network 1
        x = x.view(x.shape[0], -1)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.softmax(self.layer3(x), dim=1)
        
        return x

class Network2(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        #network2
        self.layer1 = nn.Linear(25088,5000).cuda()
        self.layer2 = nn.Linear(5000, 500).cuda()
        self.layer3 = nn.Linear(500, 2).cuda()

    def forward(self, x):
        
        #network2
        x = x.view(x.shape[0], -1)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.softmax(self.layer3(x), dim=1)
        
        return x

class Network3(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        #network3
        self.layer1 = nn.Linear(25088,5000).cuda()
        self.layer2 = nn.Linear(5000, 500).cuda()
        self.layer3 = nn.Linear(500, 2).cuda()

    def forward(self, x):
        
        #network3
        x = x.view(x.shape[0], -1)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.softmax(self.layer3(x), dim=1)
        
        return x
    
class Network4(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        #network4
        self.layer1 = nn.Linear(25088,5000).cuda()
        self.layer2 = nn.Linear(5000, 500).cuda()
        self.layer3 = nn.Linear(500, 2).cuda()

    def forward(self, x):
        
        #network4
        x = x.view(x.shape[0], -1)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.softmax(self.layer3(x), dim=1)
        
        return x
    
class Network5(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        #network5
        self.layer1 = nn.Linear(25088,5000).cuda()
        self.layer2 = nn.Linear(5000, 500).cuda()
        self.layer3 = nn.Linear(500, 2).cuda()

    def forward(self, x):
        
        #network5
        x = x.view(x.shape[0], -1)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.softmax(self.layer3(x), dim=1)
        
        return x
    
network_list = [Network1(), Network2(), Network3(), Network4(), Network5()]

class EnsembleNetwork(nn.Module):
    
    def __init__(self, model_list):
        super().__init__()
        
        self.models = model_list
        self.layer1 = nn.Linear(10,2)
        
    def forward(self, x):
        
        output = []
        
        
        for model, img in zip(self.models, x):
            output.append(model(img))
        
        output = torch.cat(output, dim=1)
        x = F.softmax(self.layer1(output), dim=1)
        
        return x

for model in model_list:
    
    model.to(device)

    for param in model.parameters():
        param.requires_grad = False

for model, network in zip(model_list, network_list):
    
    model.classifier = network

data_transforms = transforms.Compose([face_detector.FaceCropper(),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485,0.456,0.406],
                                                           std=[0.229,0.224,0.225])])

extensions = '.mp4'

data_dir = './videos'

train_dir = data_dir + '/train_sample_videos'
test_dir = data_dir + '/test_videos'

ensemble_network = EnsembleNetwork(model_list)
ensemble_network.to(device)

criterion = nn.NLLLoss()
optimizer = optim.Adam(ensemble_network.parameters(), lr=0.001)

def image_split(x):
    
    return x, x[:,:,0:112,0:112], x[:,:,0:112,112:224],\
           x[:,:,112:224,0:112], x[:,:,112:224,112:224]

epoch = 100

start = time.time()

print('Starting Training')
try:
    for e in range(epoch):

        #changes to make a random image during each epoch
        train_data = datasets.DatasetFolder(train_dir,
                                        loader=face_detector.random_frame_selector,
                                        extensions=extensions,
                                        transform=data_transforms)

        trainloader = torch.utils.data.DataLoader(train_data,
                                                  batch_size=16,
                                                  shuffle=True,
                                                  num_workers=0)

        train_losses = []
        running_loss = 0

        for images, labels in trainloader:

            labels = labels - 1
            images = images.to(device)
            labels = labels.to(device)
            splits = image_split(images)
            output = ensemble_network(splits)

            loss = criterion(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            running_loss += loss.item()
            train_losses.append(running_loss/len(trainloader))

        print(f"Training loss: {running_loss}")
except:
    pass

end = time.time()

print(f'Time taken for {epoch} epochs : {end - start}')

torch.save(ensemble_network, 'model' + str(time.time()) + '.pth')