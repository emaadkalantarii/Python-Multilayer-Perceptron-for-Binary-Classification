# Knowledge Discovery and Data Mining Project: Multilayer Perceptron Implementation

## Project Description
This project implements a Multilayer Perceptron (MLP) from scratch using Python.  It was developed as part of a Knowledge Discovery and Data Mining course. The MLP is designed for binary classification tasks.  The code includes:

* Data loading and preprocessing
* MLP architecture definition
* Forward and backward propagation
* Weight update using gradient descent
* Loss calculation
* Training loop
* Accuracy and confusion matrix calculation
* Visualization of training and validation loss

## Code Structure
The code is contained in a Jupyter Notebook (`Assignment__002_B1.ipynb`).  Here's a breakdown of the key components:

1.  **Import Libraries**:  Imports necessary Python libraries, including `numpy`, `pandas`, `matplotlib.pyplot`, and `seaborn`.
2.  **Read Data**:  Loads training and validation data from Excel files (`THA2train.xlsx`, `THA2validate.xlsx`).  *Note: These files are not included in this repository.  You will need to provide your own data in the same format.*
3.  **Data Separation**:  Separates the features (X) and labels (y) for both the training and validation datasets.  The labels are one-hot encoded using `pd.get_dummies`.
4.  **Activation Functions**:
    * `sigmoid(z)`:  Calculates the sigmoid activation function, handling potential overflow issues.
    * `sigmoid_derivative(a)`:  Calculates the derivative of the sigmoid function.
    * `softmax(z)`:  Calculates the softmax activation function for the output layer.
5.  **MLP Class**:
    * `__init__(self, input_size, hidden_size, output_size)`:  Initializes the MLP with weights and biases.  Weights are initialized to zero.
    * `forward(self, X)`:  Performs forward propagation through the network.
    * `backward(self, X, y)`:  Performs backward propagation to calculate the gradients of the weights and biases.
    * `update_weights(self, dW1, db1, dW2, db2, learning_rate)`:  Updates the weights and biases using gradient descent.
    * `compute_loss(self, y_true, y_pred)`: Computes the loss.
6.  **Data Normalization**:  Normalizes the training and validation data using mean and standard deviation.
7.  **Model Creation**:  Creates an instance of the `MLP` class.
8.  **Training**:
    * Sets hyperparameters (epochs, learning rate, batch size).
    * Iterates through epochs, performing mini-batch gradient descent.
    * Calculates and stores training and validation losses.
    * Prints training and validation losses every 100 epochs.
9.  **Evaluation**:
    * `calculate_accuracy(true_labels, predicted_labels)`:  Calculates the accuracy of the model.
    * `create_confusion_matrix(true_labels, predicted_labels, num_classes)`:  Generates the confusion matrix.
    * Calculates and prints the validation accuracy.
    * Plots the training and validation losses.
    * Displays the confusion matrix.

## How to Use the Code

1.  **Prerequisites**:  Ensure you have Python 3 installed, along with the following libraries:
    * numpy
    * pandas
    * matplotlib
    * seaborn

    You can install these using pip:
    ```bash
    pip install numpy pandas matplotlib seaborn
    ```
2.  **Data**:  Place your training and validation data in Excel files named `THA2train.xlsx` and `THA2validate.xlsx` in a directory named `DataSets`. The data should be formatted as follows:
    * The last column should contain the class labels.
    * The preceding columns should contain the features.
3.  **Run the Notebook**:  Open and run the `Assignment__002_B1.ipynb` notebook in Jupyter.  The notebook will train the MLP and display the training/validation loss curves, confusion matrix, and validation accuracy.

## Results
The notebook includes the following results:
* Plots of training and validation loss over epochs.
* A confusion matrix for the validation set.
* The validation accuracy.

## Further Improvements
Here are some potential improvements:
* Implement k-fold cross-validation.
* Add more metrics (e.g., precision, recall, F1-score).
* Include the ability to save and load trained models.
* Add more comments to the code.
* Experiment with different hyperparameters, network architectures, and activation functions.
* Extend the code to handle multi-class classification problems with more than two classes.
