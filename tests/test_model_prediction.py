import numpy as np
import torch.nn as nn
import torch
import sys
import os
sys.path.append(os.getcwd())
from inference.model_prediction import CNN, FashionMnist


model_path = "model/saved_model.pth"

def test_cnn_init():
    """
    Test the CNN initialization function by checking if the returned object is an instance of torch.nn.Module.
    This function creates an instance of the CNN class with a specified number of labels and checks if it is an instance of the nn.Module class.

    Args:
       None
    Returns:
        None
    """

    cnn = CNN(num_labels=10)
    assert isinstance(cnn, nn.Module), "CNN should be an instance of nn.Module"

def test_cnn_forward():
    """
    Test the CNN forward function by checking if the output shape is correct.
    This function creates an instance of the CNN class, generates a random input image, and passes it through the CNN. It then checks if the output shape has the expected number of classes.

    Args:
       None
    Returns:
        None
    """
    cnn = CNN(num_labels=10)
    image = np.zeros((28, 28))
    image = image.astype('float32')
    image = torch.tensor(image)
    image = image.unsqueeze(0)
    image = image.unsqueeze(1)
    image = image.float()
    output = cnn(image)
    assert output.shape[1] == 10, "Output should have 10 classes"

def test_fashion_mnist_init():
    """
    Test the FashionMnist initialization function by checking if it returns an instance of the FashionMnist class.
    This function creates an instance of the FashionMnist class with a specified model path and checks if the returned object is an instance of the FashionMnist class.

    Args:
       None
    Returns:
        None
    """
    fashion_mnist = FashionMnist(model_path)
    assert isinstance(fashion_mnist, FashionMnist), "FashionMnist should be an instance of FashionMnist"

def test_read_jpeg_image():
    """
    Test the FashionMnist.read_jpeg_image() function by checking if it returns an image with valid pixel values.
    This function creates an instance of the FashionMnist class with a specified model path and reads a JPEG image using the FashionMnist.read_jpeg_image() function. It then checks if the returned image has pixel values that are less than or equal to 255 and more than or equal to 0.

    Args:
        None

    Returns:
        None
    """
    fashion_mnist = FashionMnist(model_path)
    image = fashion_mnist.read_jpeg_image("data/inference_prediction/test_image_3.jpeg")
    assert image.max() <= 255 and image.min() >= 0, "Image values should be less than or equal to 255 and more than or equal to 0"

def test_predict():
    """
    Test the FashionMnist.predict() function by checking if it returns a valid prediction and label.
    This function creates an instance of the FashionMnist class with a specified model path and creates a test image with a white square in the middle. It then calls the FashionMnist.predict() function with this image and checks if the returned prediction is an integer and the label is a string.

    Args:
        None

    Returns:
        None
    """
    fashion_mnist = FashionMnist(model_path)
    image = np.zeros((28, 28), dtype=np.uint8)
    image[10:14, 10:14] = 255
    prediction, label = fashion_mnist.predict(image)
    assert isinstance(prediction, int), "Prediction should be an integer"
    assert isinstance(label, str), "Label should be a string"       

def test_predict_with_label():
    """
    Test the FashionMnist.predict() function with a specified label by checking if it returns a valid prediction and label.
    This function creates an instance of the FashionMnist class with a specified model path and creates a test image with a white square in the middle. It then calls the FashionMnist.predict() function with this image and a specified label and checks if the returned prediction is an integer in the range of [0, 9] and the label is a value in the list of 10 names.

    Args:
        None

    Returns:
        None
    """
    fashion_mnist = FashionMnist(model_path)
    image = np.zeros((28, 28), dtype=np.uint8)
    image[10:14, 10:14] = 255
    prediction, label = fashion_mnist.predict(image, label=3)
    names = {
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

    assert 0 <= prediction <= 9, "Prediction should be in the range of [0, 9]."
    assert label in names.values(), "Prediction label is not in the list of 10 names."