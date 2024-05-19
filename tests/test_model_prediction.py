import sys
import numpy as np
import pytest
import torch.nn as nn
import torch
import sys
import os
sys.path.append(os.getcwd())
from inference.model_prediction import CNN, FashionMnist



def test_cnn_init():
    cnn = CNN(num_labels=10)
    assert isinstance(cnn, nn.Module), "CNN should be an instance of nn.Module"

def test_cnn_forward():
    cnn = CNN(num_labels=10)
    image = np.zeros((28, 28))
    image = image.astype('float32')
    image = torch.tensor(image)
    image = image.unsqueeze(0)
    image = image.unsqueeze(1)
    image = image.float()
    print(image.shape)
    output = cnn(image)
    print(output)
    assert output.shape[1] == 10, "Output should have 10 classes"

def test_fashion_mnist_init():
    model_path = "C:\\Users\\lenovo\\Desktop\\asds_python2_project\\ASDS_python2_project\\model\\saved_model.pth"
    fashion_mnist = FashionMnist(model_path)
    assert isinstance(fashion_mnist, FashionMnist), "FashionMnist should be an instance of FashionMnist"

def test_read_jpeg_image():
    fashion_mnist = FashionMnist("C:\\Users\\lenovo\\Desktop\\asds_python2_project\\ASDS_python2_project\\model\\saved_model.pth")
    image = fashion_mnist.read_jpeg_image("C:\\Users\\lenovo\\Desktop\\asds_python2_project\\ASDS_python2_project\\data\\inference_prediction\\test_image_3.jpeg")
    assert image.max() <= 255 and image.min() >= 0, "Image values should be less than or equal to 255 and more than or euql to 0"    

# def test_plot_image():
#     fashion_mnist = FashionMnist("C:\\Users\\lenovo\\Desktop\\asds_python2_project\\ASDS_python2_project\\model\\saved_model.pth")
#     image = fashion_mnist.read_jpeg_image("C:\\Users\\lenovo\\Desktop\\asds_python2_project\\ASDS_python2_project\\data\\inference_prediction\\test_image_3.jpeg")
#     with pytest.raises(AssertionError):
#         fashion_mnist.plot_image(np.zeros((28, 28)))

def test_predict():
    model_path = "C:\\Users\\lenovo\\Desktop\\asds_python2_project\\ASDS_python2_project\\model\\saved_model.pth"
    fashion_mnist = FashionMnist(model_path)
    image = np.zeros((28, 28), dtype=np.uint8)
    image[10:14, 10:14] = 255
    prediction, label = fashion_mnist.predict(image)
    assert isinstance(prediction, int), "Prediction should be an integer"
    assert isinstance(label, str), "Label should be a string"       

def test_predict_with_label():
    model_path = "C:\\Users\\lenovo\\Desktop\\asds_python2_project\\ASDS_python2_project\\model\\saved_model.pth"
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


    assert prediction >= 0 and prediction <=9, "Prediction should be in the range of [0, 9]."
    assert label in names.values(), "Prediction label is not in the list of 10 names."