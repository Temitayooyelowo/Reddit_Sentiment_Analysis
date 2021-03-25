import requests
from datetime import datetime, timedelta
from json import loads

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
        # print(start_date)
        # print(end_date)
        print(url)
        res = requests.get(url)
        return loads(res.text)
        
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
        url = f'https://api.pushshift.io/reddit/search/comments/?q={query}&size={size}&before={end_date}&after={start_date}&subreddit={subreddits}&sort_type=score&sort=desc'
        res = requests.get(url)
        return loads(res.text)



# HOW TO USE IT
# if __name__== "__main__":
#     r = RedditAPI()
#     end = datetime.now()
#     start = end - timedelta(days=100)
    
#     # print(start.strftime('%s'))
#     # print(end.strftime('%s'))
#     print(r.search_for_posts(f'GME', str(1000), start.strftime('%s'), end.strftime('%s'), f'wallstreetbets'))