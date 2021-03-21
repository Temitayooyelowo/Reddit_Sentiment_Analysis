import numpy as np
import pandas as pd
import nltk
import re
import emoji

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from emot.emo_unicode import EMOTICONS
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn import metrics
from sklearn.linear_model import LogisticRegression


class TextPreprocessing:
  def __init__(self):
    self.lemmatizer = WordNetLemmatizer()

  '''
  https://www.kite.com/python/answers/how-to-make-a-pandas-dataframe-string-column-lowercase-in-python#:~:text=Use%20str.,%5B%22first_column%22%5D%20lowercase.
  '''
  def convert_to_lower_case(self, pd_column):
    return pd_column.str.lower()

  def remove_punctuation_from_string(self, pd_column):
    return pd_column.str.replace('[^\w\s]','')

  def remove_url(self, pd_column):
    return pd_column.str.replace('https?:\/\/\S+|www.\S+', '')

  def remove_usernames_and_subreddit_names(self, pd_column):
    return pd_column.str.replace('u\/\S+|@\S+|r\/\S+', '')

  def convert_emojis_to_english(self, text):
    return emoji.demojize(text).replace(':','')

  # Converting emoticons to words
  def convert_emoticons(self, text):
    for emot in EMOTICONS:
      text = re.sub(u'('+emot+')', "_".join(EMOTICONS[emot].replace(",","").split()), text)
    return text

  def lemmatize_words(self, text):
    return lemmatizer.lemmatize(text)

  def clean_data(self, pd_column):
    new_column = self.convert_to_lower_case(pd_column)
    new_column = self.remove_usernames_and_subreddit_names(new_column)
    new_column = self.remove_url(new_column)

    new_column = new_column.apply(self.convert_emojis_to_english)
    new_column = new_column.apply(self.convert_emoticons)
    new_column = new_column.apply(self.lemmatize_words)

    new_column = self.remove_punctuation_from_string(new_column)
    return new_column


df = pd.read_csv('./dataset/Reddit_Data.csv', sep=',')
x = df['clean_comment'].astype('U')

text_preprocessing = TextPreprocessing()
new_column = text_preprocessing.clean_data(x)

print(new_column.head())

# nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
y = df.values[:,1].astype('int')

'''
  we use the count vectorizer to vectorize based on frequency of occurrence which favours words that occur more frequently
'''
# vectorizer = TfidfVectorizer(stop_words=stop_words, max_df=0.6, use_idf=True, strip_accents='ascii')
# vectorizer = HashingVectorizer(stop_words=stop_words, lowercase=True, strip_accents='ascii')
vectorizer = CountVectorizer(stop_words=stop_words, lowercase=True, binary = True, strip_accents='ascii')

X = vectorizer.fit_transform(new_column)

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

# 0.6281879194630873 count vectorizer
# 0.5562416107382551 tfidf vectorizer

'''
  Logistic Regression Prediction
'''
log_regression_classifier = LogisticRegression(max_iter=10000)

log_regression_classifier.fit(X_train, y_train)
y_pred_nb = log_regression_classifier.predict(X_test)

print('\n_________________________________Logistic Regression_________________________________')
print(metrics.classification_report(y_test, y_pred_nb))
print('Accuracy: ', metrics.accuracy_score(y_test, y_pred_nb))
# 0.854675615212528 # count vectorizer
# 0.821744966442953 # tfidf vectorizer
# 0.8297091722595078 # hashing vectorizer