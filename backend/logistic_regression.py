import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import metrics
import time
from gensim.parsing.preprocessing import remove_stopwords
from sklearn.feature_extraction import text
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer


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
    def __init__(self):
        self.vectorizer = CountVectorizer(binary = True, strip_accents='ascii', tokenizer=LemmaTokenizer())
        self.k_count = {}
        self.theta_k = {}
        self.theta_j_k = {}

    def fit(self, X, Y) :
        start = time.time()

        X = self.vectorizer.fit_transform(X)
        X_vector_train = X.toarray()

        # count number of occurences of class vlues in train set
        unique, counts = np.unique(Y, return_counts=True)
        self.k_count = dict(zip(unique, counts))
        self.column_names = unique

        for k in self.k_count :
            # theta_k
            self.theta_k[k] = self.k_count[k]/float(Y.shape[0])

            # theta_j_k[k]
            self.theta_j_k[k] = []

            # indexes for samples that are in k subreddit
            k_indexes = np.where(Y == k)[0]

            # filter X_vector_train with k_indexes
            filtered_X = np.take(X_vector_train, k_indexes, axis=0)
            # sum of all j/word binary occurences in a k/subreddit
            filtered_X_sum = filtered_X.sum(axis=0)

            # for every word (j), building self.theta_j_k[k][j]
            for j in range(X_vector_train.shape[1]) :
                # NO LAPLACE SMOOTHING
                # prob = filtered_X_sum[j]/float(self.k_count[k])

                # LAPLACE SMOOTHING
                prob = (filtered_X_sum[j] + 1)/(float(self.k_count[k]) + 2)

                self.theta_j_k[k].append(prob)

        end = time.time()
        print("Fit time:", (end - start))

        return self.theta_k, self.theta_j_k

    def predict(self, X_test):
        X_test = self.vectorizer.transform(X_test)
        X_test = X_test.toarray()
        num_classes, num_features = X_test.shape

        return self.__predict(X_test, num_classes, num_features)

    def __predict(self, X_test, num_classes, num_features):
        class_prob = [[] for _ in range(num_classes)]

        for k in self.k_count:
            feature_likelihood = 0
            for j in range(num_features):
                X_j = X_test[:,j]
                feature_likelihood += X_j * np.log(self.theta_j_k[k][j]) + (
                    1 - X_j) * np.log(1 - self.theta_j_k[k][j])
            curr_class_prob = feature_likelihood + np.log(self.theta_k[k])

            for idx, val in enumerate(curr_class_prob):
                class_prob[idx].append(val)
                if len(class_prob[idx]) == len(self.column_names):
                    column_idx = np.argmax(class_prob[idx])
                    class_prob[idx] = self.column_names[column_idx]

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

    # clean dataset
    X = clean_data(X)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, test_size=0.2)

    print("Train x:", X_train.shape, "y:", Y_train.shape)
    print("Test x:", X_test.shape, "y:", Y_test.shape)

    bnb = Naive_Bayes()

    bnb.fit(X=X_train, Y=Y_train)
    y_pred = bnb.predict(X_test)

    bnb.report_accuracy(y_pred, Y_test)