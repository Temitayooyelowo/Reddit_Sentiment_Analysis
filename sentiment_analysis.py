import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords

# changing from ENGLISH_STOP_WORDS to nltk.corpus.stopwords increases the accuracy with LogisticRegression by about 3%
# and increases the accuracy for multinomial naive bayes by about 3%
curr_stop_words = set(stopwords.words('english'))

'''
train.values is a 2d array of
[
  ['comment', sentiment]
]
sentiment: -1 is negative, 0 is neutral, and 1 is positive
'''
df = pd.read_csv('./dataset/Reddit_Data.csv', sep=',')
# print(f'Sentences: {df.values[:,0]}')
# print(f'Sentiment: {df.values[:,1]}')

x = df.values[:,0].astype('U')
y = df.values[:,1].astype('int')


# use TfidfVectorizer to remove words that occur in more than 80% of features
# use strip_accents to ignore non-english words https://stackoverflow.com/a/57286757
# vectorizer = TfidfVectorizer(stop_words=ENGLISH_STOP_WORDS.union(['book']), lowercase=True, max_df=0.7, use_idf=True, strip_accents='ascii')
vectorizer = CountVectorizer(stop_words=curr_stop_words, lowercase=True, binary = True, strip_accents='ascii')

X = vectorizer.fit_transform(x)

# split data such that we train our model on 70% of the data and test on 30%
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)



'''
  Naive Bayes Prediction
'''
mnb_classifier = MultinomialNB()
# mnb_classifier = BernoulliNB()

mnb_classifier.fit(X_train, y_train)
y_pred_nb = mnb_classifier.predict(X_test)

print('\n_________________________________Multionmial Naive Bayes_________________________________')
print(metrics.classification_report(y_test, y_pred_nb))
print('Accuracy: ', metrics.accuracy_score(y_test, y_pred_nb))



'''
  Logistic Regression Prediction
'''
log_regression_classifier = LogisticRegression(max_iter=10000)

log_regression_classifier.fit(X_train, y_train)
y_pred_nb = log_regression_classifier.predict(X_test)

print('\n_________________________________Logistic Regression_________________________________')
print(metrics.classification_report(y_test, y_pred_nb))
print('Accuracy: ', metrics.accuracy_score(y_test, y_pred_nb))


