import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
from sentiment_analysis import TextPreprocessing, LemmaTokenizer

class AnalyzeDataset:
    def __init__(self):
        pass

    '''
        Function to get the data from the csv
        The csv data will be used to train the model
    '''
    def initialize_dataset(self):
        file_names = ['test_preprocessing.csv', 'stock_data.csv', 'stock_sentiment.csv', 'Project6500.csv']
        # file_names = ['test_preprocessing.csv']
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
class Naive_Bayes:
    def __init__(self, alpha=1):
        self.alpha = alpha
        self.vectorizer = CountVectorizer(binary = True, strip_accents='ascii', tokenizer=LemmaTokenizer())
        self.theta_k = {}
        self.theta_k_j = {}
        self.class_count_dict = {}

    def fit(self, X_train, Y_train) :
        X_train = self.vectorizer.fit_transform(X_train)
        X_train = X_train.toarray()

        # count number of occurences of class vlues in train set
        unique, counts = np.unique(Y, return_counts=True)
        self.class_count_dict = dict(zip(unique, counts))
        self.class_names = unique

        for k in self.class_count_dict:
            # theta_k
            self.theta_k[k] = self.class_count_dict[k]/float(Y.shape[0])
            self.theta_k_j[k] = []

            num_examples_y_equals_k = np.take(X_train, np.where(Y_train == k)[0], axis=0)
            total_num_of_examples = num_examples_y_equals_k.sum(axis=0)

            for j in range(X_train.shape[1]) :
                # use laplace estimation to handle zero probabilities
                prob = (total_num_of_examples[j] + self.alpha)/(float(self.class_count_dict[k]) + 2)
                self.theta_k_j[k].append(prob)

        return self.theta_k, self.theta_k_j

    def predict(self, X_test):
        X_test = self.vectorizer.transform(X_test)
        X_test = X_test.toarray()
        num_classes, num_features = X_test.shape

        return self.__predict(X_test, num_classes, num_features)

    def __predict(self, X_test, num_classes, num_features):
        class_prob = [[] for _ in range(num_classes)]

        for k in self.class_count_dict:
            feature_likelihood = 0
            for j in range(num_features):
                X_j = X_test[:,j]
                feature_likelihood += X_j * np.log(self.theta_k_j[k][j]) + (
                    1 - X_j) * np.log(1 - self.theta_k_j[k][j])
            self.__class_prob(class_prob, feature_likelihood + np.log(self.theta_k[k]))

        return class_prob

    def __class_prob(self, class_prob, curr_class_prob):
        for i, val in enumerate(curr_class_prob):
            class_prob[i].append(val)
            if len(class_prob[i]) == len(self.class_names):
                class_prob[i] = self.class_names[np.argmax(class_prob[i])]

        return class_prob

    def report_accuracy(self, y_pred, y_test):
        print(metrics.classification_report(y_test, y_pred))
        print('Accuracy: ', metrics.accuracy_score(y_test, y_pred))

def clean_data(x):
    text_preprocessing = TextPreprocessing()
    clean_column = text_preprocessing.clean_data(x)
    print(clean_column.head())
    return clean_column

if __name__ == "__main__" :
    analyze_dataset = AnalyzeDataset()
    X, Y = analyze_dataset.initialize_dataset()
    X = clean_data(X)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.7, random_state=42)

    print("Train x:", X_train.shape, "y:", Y_train.shape)
    print("Test x:", X_test.shape, "y:", Y_test.shape)

    naive_bayes = Naive_Bayes()

    naive_bayes.fit(X_train, Y_train)
    y_pred = naive_bayes.predict(X_test)

    naive_bayes.report_accuracy(y_pred, Y_test)