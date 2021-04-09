import time
from flask import Flask, request, jsonify
from sentiment_analysis import SentimentAnalysis
from flask_cors import CORS
from pushshift import RedditAPI
from datetime import datetime, timedelta
import time
import pandas as pd

app = Flask(__name__)
reddit = RedditAPI()
sentiment = SentimentAnalysis()
date_format = "%m/%d/%Y %H:%M:%S"
CORS(app)
time_options = {
	"past year": 365,
	"past 3 months": 90,
	"past month": 30,
	"past week": 7
}

@app.route('/time')
def get_current_time():
    return {'time': time.time()}

@app.route('/')
def index():
	return 'Hello, Flask!'

'''
Expecting in body: 
{
	time: "past year" or "past 3 months" or "past month" or "past week",
	ticker: "some ticker"
}
'''

@app.route('/sentiment', methods=['GET'])
def sentiment():
	body = dict(request.get_json())

	# convert time to EPOCH time
	now = datetime.now()
	start = now - timedelta(days=time_options[body['time']])
	
	end_epoch = int(time.mktime(time.strptime(f'{now.month}/{now.day}/{now.year} {now.hour}:{now.minute}:{now.second}', date_format)))
	start_epoch = int(time.mktime(time.strptime(f'{start.month}/{start.day}/{start.year} {start.hour}:{start.minute}:{start.second}', date_format)))

	# which would give more meaningful results, comments or posts?
	comments = reddit.search_for_comments(body['ticker'], 1000, start_epoch, end_epoch, 'wallstreetbets')

	results = get_sentiment({'body': comments})
	print(results)
	return jsonify({'sentiments': results.tolist()})

def get_sentiment(data):
	df = pd.DataFrame(data=data)

	x = df['body'].astype('U')
	clean_column = sentiment.clean_data(x)
	X = sentiment.vectorizer.transform(clean_column)

	y_pred_nb = sentiment.logistic_regression_predict(X)

	return y_pred_nb

sentiment = SentimentAnalysis()
if __name__ == '__main__':
	sentiment.train_model()
	app.run(debug=False)