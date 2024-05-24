# ASDS Python2 Project

This is the final project for the ASDS faculty's Python2 course. The project aims to create a model using PyTorch, write inference code for prediction, make an API call, and run tests. The dataset used is the FashionMNIST data, which includes 10 types of 28x28 fashion images.

## Fashion-MNIST Dataset

Fashion-MNIST is a dataset of Zalando's article images—consisting of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes. We intend Fashion-MNIST to serve as a direct drop-in replacement for the original MNIST dataset for benchmarking machine learning algorithms. It shares the same image size and structure of training and testing splits.

The labels are as follows:

- 0: "T-shirt/top"
- 1: "Trouser"
- 2: "Pullover"
- 3: "Dress"
- 4: "Coat"
- 5: "Sandal"
- 6: "Shirt"
- 7: "Sneaker"
- 8: "Bag"
- 9: "Ankle boot"

Data link `https://github.com/zalandoresearch/fashion-mnist/tree/master/data/fashion`
Create data folder and put it there.

## Project Structure

- The test images for API predictions are separated in the `model_creation/separating_test_images.ipynb` notebook.
- Data preprocessing, analysis, and modeling are done in the `model_creation/data_processing_modelling.ipynb` notebook.
- The final model reading and predicting class with functions are written in the `inference/model_prediction.py` file.
- An experiment on one image reading, plotting, and predicting is done in the `inference_model_predictions_on_examples.ipynb` notebook.
- The trained model is saved in the `model/saved_model.pth` file.
- Tests for the above classes and functions are written in the `tests/test_model_prediction.py` file. To run tests, go to the `ASDS_python2_project` folder by terminal and run `pytest tests/test_model_prediction.py` command.
- The API call is written in the `main.py` function. Running it, you will be able to upload a .jpeg image by web, which will predict and print the output of the image label.

## How to Run

To start the project, run the command `python main.py`.