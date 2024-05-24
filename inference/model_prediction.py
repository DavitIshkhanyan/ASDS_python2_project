import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn as nn


class CNN(nn.Module):
    """
    CNN class is used as model architecure for Fashion Mnist dataset.
    It's Neural network, which will load model weights and use for predictions.

    ...

    Attributes
    ------------
    num_labels(int): 
        Integer number for the count of distinct labels in dataset(default value is 10).

    Methods
    ------------
    forward(x): 
        Forwarding input via neural network and gets the predicted vector.
    """

    def __init__(self, num_labels=10):
        """
        Args:
            num_labels(int): The number of labels in model prediction.

        """

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
        """
        Forwarding input via neural network and gets the predicted vector.

        Args:
            x(torch.tensor): The tensor which is the image 1x1x28x28 size.
        
        Returns (torch.tensor):
            Output vector after neural network.
        """
        x = self.layers(x)
        return x
   

 
class FashionMnist:
    """
    Fashion Mnsit class is for Fashion Mnist dataset for reading jpeg
    images, plotting and predicting image labels.

    ...

    Attributes
    ------------
    model_path(str):  
        Path to the saved model.
    device(torch.device): 
        Device on which model will predict. 
        If exists cuda then will take it, otherwsie cpu.
    model(CNN):  
        Object of CNN class, which is main model.
    fashion_dict(dict):  
        Class numbers and lables mapping dictionay.
        The dictionary size is 10, meaning that have 10 unique labels.

    Methods
    ------------
    read_jpeg_image(path):
        Reading jpef format saved image into array.
    plot_image(image):
        Plotting fshion mnist image from input.
    predict(image, label=None):
        Predicting image label and returing back with class
        number and class label.
        If label is passed then printing both real and predicted labels.
        If None, then not printing.
    """
    def __init__(self, model_path):
        """
        Args:
            model_path(str): Saved model path.

        """
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
        """
        Reading .jpeg format saved image inot array.

        Args:
            path(str): 
                Path to image file.
        
        Returns:
            Image array.
        """
        image_read = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        return image_read

    def plot_image(self, image):
        """
        Plotting image from array.

        Args:
            image(array): 
                Image array.
        """
        plt.figure(figsize=(2, 2))
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image, cmap=plt.cm.binary)
        plt.show()
    
    def predict(self, image, label=None):
        """
        Predicts image label by model.

        Args:
            image(array): 
                Image array.
            label(str): 
                Real label of image.
                If label is None nothing happens, otherwise prints real and predicted labels of image.
        
        Returns (tuple):
            Predicted image label number, and label name.
        """
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
        
        