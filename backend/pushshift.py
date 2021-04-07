from datetime import datetime, timedelta
from json import loads
# from pmaw import PushshiftAPI
import time
import requests



# api = PushshiftAPI()

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
        url = f'https://api.pushshift.io/reddit/search/submission/?q={query}&limit={size}&before={end_date}&after={start_date}&subreddit={subreddits}&sort_type=num_comments&sort=desc'
        res = requests.get(url).json()
        list_of_comments = []
        # res = list(api.search_submissions(subreddit=subreddits, limit=size, before=end_date, after=start_date, sort='desc', sort_type='score'))

        for post in res['data']:
            post_id = post['permalink'].split('/')[4]
            print(post_id)
            res_comments = requests.get(f'https://api.pushshift.io/reddit/search/comment/?link_id={post_id}&size=100&sort_type=score&sort=desc').json()
            list_of_comments.extend([comment['body'] for comment in res_comments['data'] if comment['body'] != '[removed]'])
            
        
        return list_of_comments
        
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
        res = requests.get(url).json()

        # res = list(api.search_comments(subreddit=subreddits, limit=size, before=end_date, after=start_date, sort='desc', sort_type='score'))
        
        return [comment['body'] for comment in res['data']]



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
#     t1 = time.time()
#     res = r.search_for_comments(f'GME', 500, start_ep, end_ep, f'wallstreetbets')
#     t2 = time.time()
#     print(res)
#     print(len(res))
#     print(f'\n\nTIME ELAPSED: {t2 - t1}s\n\n')