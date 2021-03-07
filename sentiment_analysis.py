import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn import metrics

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

'''
(37250,)
(37250,)
'''
x = df.values[:,0].astype('U')
y = df.values[:,1].astype('int')

# use TfidfVectorizer to remove words that occur in more than 80% of features
# use strip_accents to ignore non-english words https://stackoverflow.com/a/57286757
curr_stop_words = ENGLISH_STOP_WORDS.union(["book"])
# vectorizer = TfidfVectorizer(stop_words=curr_stop_words, lowercase=True, max_df=0.8, use_idf=True, strip_accents='ascii')
vectorizer = TfidfVectorizer(stop_words=curr_stop_words, lowercase=True, max_df=0.7, use_idf=True, strip_accents='ascii')
X = vectorizer.fit_transform(x)

# print(vectorizer.get_feature_names()[36575])
# print(vectorizer.get_feature_names()[10090])
# print(vectorizer.get_feature_names()[29349])
# print(vectorizer.get_feature_names()[27086])
# print(vectorizer.get_feature_names()[49332])
# print(vectorizer.get_feature_names()[36774])
# print(vectorizer.get_feature_names()[51403])
# print(vectorizer.get_feature_names()[48694])
# print(vectorizer.get_feature_names()[38893])

# split data such that we train our model on 70% of the data and test on 30%
# X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.6, random_state=42)

# print(X_train.shape)
# print(y_train.shape)
# print(y_train.shape)
# print(y_test.shape)

# print(X_train[0])

mnb_classifier = MultinomialNB()
# mnb_classifier = BernoulliNB()

mnb_classifier.fit(X_train, y_train.astype('int'))
y_pred_nb = mnb_classifier.predict(X_test)

print(y_pred_nb)
print(y_test)
# print(mnb_classifier.predict(X_train[0]))
print('\n___________________________________________')
print(metrics.classification_report(y_test, y_pred_nb))
# print('\n___________________________________________')
# print(metrics.roc_auc_score(y_test, mnb_classifier.predict_proba(X_test)))

# count = 0
# for i in range(0, len(y_test)):
#   count = count + 1 if y_test[i] == y_pred_nb[i] else count + 0

# print(count / len(y_test))