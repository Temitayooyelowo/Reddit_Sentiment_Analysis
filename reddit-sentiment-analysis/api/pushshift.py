import requests
from datetime import datetime, timedelta
from json import loads
import time

class RedditAPI:
    def __init__(self):
        pass

    '''
    inputs:
        query: Query term for comments and submission
        size: number of results to recieve (does not always give the result you want)
        start_date: Restrict results to those made after this epoch time
        end_date: Restrict results to those made before this epoch time
        subreddits: Restrict results to subreddit (comma delimited for multiples)

    output: dictionary of the results
    notes: sorts results by the reddit score in descending order
    '''

    def search_for_posts(self, query, size, start_date, end_date, subreddits):
        url = f'https://api.pushshift.io/reddit/search/submission/?q={query}&size={size}&before={end_date}&after={start_date}&subreddit={subreddits}&sort_type=score&sort=desc'
        res = loads(requests.get(url).text)
        
        list_of_post_titles = []
        for post in res['data']:
            list_of_post_titles.append(post['title'])

        return list_of_post_titles
        
    '''
    inputs:
        query: Query term for comments and submission
        size: number of results to recieve (does not always give the result you want)
        start_date: Restrict results to those made after this epoch time
        end_date: Restrict results to those made before this epoch time
        subreddits: Restrict results to subreddit (comma delimited for multiples)
        
    output: dictionary of the results
    notes: sorts results by the reddit score in descending order
    '''
    def search_for_comments(self, query, size, start_date, end_date, subreddits):
        url = f'https://api.pushshift.io/reddit/search/comment/?q={query}&size={size}&before={end_date}&after={start_date}&subreddit={subreddits}&sort_type=score&sort=desc'
        res = loads(requests.get(url).text)
        
        list_of_comments = []

        for post in res['data']:
            list_of_comments.append(post['body'])

        return list_of_comments



# if __name__== "__main__":
#     r = RedditAPI()
#     format = "%m/%d/%Y %H:%M:%S"
#     now = datetime.now()
#     start = now - timedelta(days=100)
    
#     end_ep = int(time.mktime(time.strptime(f'{now.month}/{now.day}/{now.year} {now.hour}:{now.minute}:{now.second}', format)))
#     start_ep = int(time.mktime(time.strptime(f'{start.month}/{start.day}/{start.year} {start.hour}:{start.minute}:{start.second}', format)))

#     # print(start.strftime('%s'))
#     print(end_ep)
#     print(start_ep)
#     print(r.search_for_comments(f'GME', 10, start_ep, end_ep, f'wallstreetbets'))