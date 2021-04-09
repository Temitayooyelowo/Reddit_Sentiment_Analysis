import numpy as np
import pandas as pd
import nltk
import re
import emoji

from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from emot.emo_unicode import UNICODE_EMO, EMOTICONS
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

# Uncomment if you haven't installed stopwords, punkt, and wordnet
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')

def create_stop_words():
  file_names = ['nasdaq_screener_1616787733047.csv', 'nasdaq_screener_1616787777844.csv', 'nasdaq_screener_1616787801745.csv']
  # list comprehension performs better in terms of performance since we don't need to append to the array each time
  frames = [pd.read_csv(f'../dataset/tickers/{file_name}', sep=',', usecols= ['Symbol']) for file_name in file_names]

  df = pd.concat(frames, ignore_index=True)
  stock_tickers = df['Symbol'].str.lower()

  series_set = set(stock_tickers)
  # stop_words = set(stopwords.words('english'))
  # return series_set.union(stop_words)

  return series_set

stop_words = create_stop_words()

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
    return emoji.demojize(text).replace(':',' ')

  # Converting emoticons to words
  def convert_emoticons(self, text):
    for emot in EMOTICONS:
      text = re.sub(u'('+emot+')', "_".join(EMOTICONS[emot].replace(",","").split()), text)
    return text

  def lemmatize_words(self, text):
    return lemmatizer.lemmatize(text)

  def stop_words(self, text):
    return " ".join([word for word in str(text).split() if word not in stop_words])

  def clean_data(self, pd_column):
    new_column = self.convert_to_lower_case(pd_column)
    new_column = self.remove_usernames_and_subreddit_names(new_column)
    new_column = self.remove_url(new_column)

    new_column = new_column.apply(self.convert_emojis_to_english)
    new_column = new_column.apply(self.convert_emoticons)
    new_column = self.remove_punctuation_from_string(new_column)
    new_column = new_column.apply(self.stop_words)
    # new_column = new_column.apply(self.lemmatize_words)

    return new_column

class LemmaTokenizer(object):
  def __init__(self):
    self.wnl = WordNetLemmatizer()
  def __call__(self, doc):
    return [self.wnl.lemmatize(t) for t in word_tokenize(doc) if t not in stop_words]


class SentimentAnalysis:
  def __init__(self):
    pass

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

  def train_model(self):
    x, Y = self.initialize_dataset()
    clean_column = self.clean_data(x)
    X = self.setup_vectorizer(clean_column)
    X_train, X_test, y_train, y_test = self.split_into_test_and_train(X, Y)

    # Naive Bayes
    self.naive_bayes_train(X_train, y_train)
    naive_bayes_sentiments = self.naive_bayes_predict(X_test)
    self.report_accuracy(naive_bayes_sentiments, y_test, 'Naive Bayes')

    # Logistic Regression
    self.logistic_regression_train(X_train, y_train)
    logistic_regression_sentiments = self.logistic_regression_predict(X_test)
    self.report_accuracy(logistic_regression_sentiments, y_test, 'Logistic Regression')

  def clean_data(self, x):
    text_preprocessing = TextPreprocessing()
    clean_column = text_preprocessing.clean_data(x)
    print(clean_column.head())
    return clean_column

  def setup_vectorizer(self, clean_column):
    '''
      we use the count vectorizer to vectorize based on frequency of occurrence which favours words that occur more frequently
    '''
    # vectorizer = HashingVectorizer(stop_words=stop_words, lowercase=True, strip_accents='ascii')
    # vectorizer = TfidfVectorizer(stop_words=stop_words, max_df=0.6, use_idf=True, strip_accents='ascii')
    self.vectorizer = CountVectorizer(binary = True, strip_accents='ascii', tokenizer=LemmaTokenizer())

    X = self.vectorizer.fit_transform(clean_column)
    return X

  def split_into_test_and_train(self, X, Y):
    return train_test_split(X, Y, train_size=0.7, random_state=42)

  def logistic_regression_train(self, X_train, y_train):
    self.log_regression_classifier = LogisticRegression(max_iter=10000)
    self.log_regression_classifier.fit(X_train, y_train)

  def logistic_regression_predict(self, X_test):
    return self.log_regression_classifier.predict(X_test)
    # 0.854675615212528 # count vectorizer
    # 0.821744966442953 # tfidf vectorizer
    # 0.8297091722595078 # hashing vectorizer

  def naive_bayes_train(self, X_train, y_train):
    self.naive_bayes_classifier = MultinomialNB()
    # mnb_classifier = BernoulliNB()

    self.naive_bayes_classifier.fit(X_train, y_train)

  def naive_bayes_predict(self, X_test):
    return self.naive_bayes_classifier.predict(X_test)
    # 0.6281879194630873 count vectorizer
    # 0.5562416107382551 tfidf vectorizer

  def report_accuracy(self, y_pred_nb, y_test, model_name):
    print(f'\n_________________________________{model_name}_________________________________')
    print(metrics.classification_report(y_test, y_pred_nb))
    print('Accuracy: ', metrics.accuracy_score(y_test, y_pred_nb))

if __name__ == "__main__":
  sentiment_analysis = SentimentAnalysis()
  sentiment_analysis.train_model()

  test = {
    'body': [
      'I want to buy PD!! PD to the moon!', # 1
      'I hit my head on a pole and became more retarded', # 0
      'I used my kids tuition to buy BB, my wife found out and filed for divorce!', #-1
      '🚀', #1
    ]
  }
  df = pd.DataFrame(data=test)
  print(df.head(), '\n')

  x = df['body'].astype('U')
  clean_column = sentiment_analysis.clean_data(x)
  X = sentiment_analysis.vectorizer.transform(clean_column)

  y_pred_nb = sentiment_analysis.naive_bayes_predict(X)
  print(f'Naive Bayes Sentiment: {y_pred_nb}')

  y_pred_nb = sentiment_analysis.logistic_regression_predict(X)
  print(f'Logistic Regression Sentiment: {y_pred_nb}')

  '''
    prediction is currently [1 0 1 1] instead of [1 0 -1 1]
  '''
