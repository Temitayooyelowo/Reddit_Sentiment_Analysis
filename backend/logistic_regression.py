from math import loglp
import numpy as np
import pandas as pd

class AnalyzeDataset:
    def __init__(self):
        pass

    '''
        Function to get the data from the csv
        The csv data will be used to train the model
    '''
    def initialize_dataset(self):
        # file_names = ['test_preprocessing.csv', 'Reddit_Data.csv']
        file_names = ['test_preprocessing.csv']
        # list comprehension performs better in terms of performance since we don't need to append to the array each time
        frames = [pd.read_csv(f'../dataset/{file_name}', sep=',') for file_name in file_names]
        df = pd.concat(frames, ignore_index=True)
        print('\n', df.head())
        x = df['clean_comment'].astype('U')
        y = df.values[:,1].astype('int')

        return x, y
'''
    Class to calculate the logistic regression using gradient descent
'''
class LogisticRegression:
    def __init__(self, learn_rate, tolerance):
        self.learn_rate = learn_rate
        self.tolerance = tolerance

    def __sigmoid(self, x):
        1 / (1 + np.exp(-x))

    def train(self, tolerance):
        pass

    def fit(self, x, y):
        num_features, num_samples = X.shape

        # create an array with 1 row and num_features column
        weights = np.zeros((num_features, 1))
        bias = np.ones((num_samples, 1))

        k = 0
        while True:
            delta = np.zeros((num_features+1, 1))
            # loop through num of samples 
            for i in range(num_samples):
                X = np.matrix() # TODO: need to finish this
                true_output = y[i]

                # use dot product to calculate curr delta
                predicted_output =  self.__sigmoid(np.dot(np.transpose(weight, X)))
                y_minus_sigmoid = np.subtract(true_output, predicted_output)
                curr_delta = np.dot(X, y_minus_sigmoid)

                delta += curr_delta

            delta = delta * -1 
            # we might need to updated learning rate for each iteration
            updated_weights = np.subtract(weights, np.dot(learn_rate, delta))
            norm = np.subtract(updated_weights, weights)
            norm = np.linalg.norm(norm)
            norm = np.square(norm)

            # should converge when norm is less than tolerance
            if norm < self.tolerance:
                accuracy = None
                # TODO: should make a prediction and compare that with the expected value 
                # might need to create a function for this 
                return accuracy 
            k += 1

    def accuracy_evaluation(self):
        pass

    def predict(self):
        pass
    