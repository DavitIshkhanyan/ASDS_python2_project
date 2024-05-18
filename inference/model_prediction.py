import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


class CNN(nn.Module):
    def __init__(self, num_labels=10):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),  # 26 x 26 x 32
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=3),  # 24 x 24 x 32
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 12 x 12 x 32
            nn.Conv2d(32, 64, kernel_size=3),  # 10 x 10 x 64
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3),  # 8 x 8 x 64
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 4 x 4 x 64
            nn.Flatten(),
            nn.Linear(in_features=1024, out_features=512),
            nn.ReLU(),
            nn.Linear(512, num_labels),
        )

    def forward(self, x):
        x = self.layers(x)
        return x
   

 
class FashionMnist:
    def __init__(self, model_path):
        self.fashion_dict = {
            0: "T-shirt/top",
            1: "Trouser",
            2: "Pullover",
            3: "Dress",
            4: "Coat",
            5: "Sandal",
            6: "Shirt",
            7: "Sneaker",
            8: "Bag",
            9: "Ankle boot",
            }
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CNN(num_labels=10)
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)

    def read_jpeg_image(self, path):
        image_read = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        return image_read

    def plot_image(self, image):
        plt.figure(figsize=(2, 2))
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image, cmap=plt.cm.binary)
        plt.show()
    
    def predict(self, image, label=None):
        preprocessed_image = image.astype('float32')
        preprocessed_image = torch.tensor(preprocessed_image)
        preprocessed_image = preprocessed_image.unsqueeze(0)
        preprocessed_image = preprocessed_image.unsqueeze(1)

        preprocessed_image = preprocessed_image.float().to(self.device)
        prob_prediction = self.model(preprocessed_image)
        _, prediction = torch.max(prob_prediction, 1)
        prediction = prediction.item()
        
        if label is not None:
            print('Real label of image is: ', f'{self.fashion_dict[label]} ({label})')
            print('Predicted label of image is: ', f'{self.fashion_dict[prediction]} ({prediction})')
        
        return prediction, self.fashion_dict[prediction]
        
        