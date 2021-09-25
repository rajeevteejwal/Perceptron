import numpy as np
import logging
from tqdm import tqdm

class perceptron:
    def __init__(self, eta, epochs):
        self.weights = np.random.randn(3) * 1e-4
        self.epochs = epochs
        self.eta = eta
        logging.info(f"These are initial weights before training {self.weights}")

    def activationFunction(self, inputs,weights):
        z = np.dot(inputs, weights)
        return np.where(z > 0, 1, 0)

    def fit(self, X, y):
        self.X = X
        self.y = y
        # here X will have all data
        x_with_bias = np.c_[self.X,-np.ones((len(self.X),1))]
        logging.info(f"X_with_bias : \n{x_with_bias}")

        for epoch in tqdm(range(self.epochs), total=self.epochs, desc="training the model" ):
            logging.info("--"*10)
            logging.info(f"For epoch : {epoch}")
            logging.info("--"*10)
            y_hat = self.activationFunction(x_with_bias,self.weights)
            logging.info(f"Predicted value after forward pass :\n {y_hat} ")
            self.error = self.y - y_hat
            logging.info(f"Error : \n {self.error}")
            self.weights = self.weights + self.eta * np.dot(x_with_bias.T, self.error)
            logging.info(f"Updated weights after epoch : \n{epoch}/{self.epochs} :\n{self.weights}")
            logging.info("##"*10)

    def predict(self,x):
        logging.info(f"predict function: Input : {x}")
        print(f"predict function: Input : {x}")
        x_with_bias = np.c_[x,-np.ones((len(x),1))]
        logging.info(f"predict function: x_with_bias : {x_with_bias}")
        print(f"predict function: x_with_bias : {x_with_bias}")
        z = self.activationFunction(x_with_bias,self.weights)
        return z

    def total_loss(self):
        total_loss = np.sum(self.error)
        logging.info(f"Total loss : \n{total_loss}")
        return total_loss

# perceptron has input dimension
# perceptron has weights
# perceptron has bias

