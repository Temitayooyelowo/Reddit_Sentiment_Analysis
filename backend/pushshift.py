from datetime import datetime, timedelta
from json import loads
from threading import Thread
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
        list_of_comments = [] #lists are thread safe so we don't need a lock
        jobs = []

        url = f'https://api.pushshift.io/reddit/search/submission/?q={query}&limit={size}&before={end_date}&after={start_date}&subreddit={subreddits}&sort_type=score&sort=desc'
        res = requests.get(url).json()

        for post in res['data']:
            post_id = post['permalink'].split('/')[4]
            thread = Thread(target=self.search_for_comments_by_post_id, args=(post_id, list_of_comments))
            jobs.append(thread)
            thread.start()

        # tells main thread to wait for threads to complete before continuing
        for job in jobs:
            job.join()

        return list_of_comments

    '''
    inputs:
        post_id: the id that uniquely identifies a post on Reddit, this is used to get the comments regarding a specific post
        list_of_comments: a list that contains the comments from Reddit

    output: does not return any output since arrays are passed by references
    notes: sorts results by the reddit score in descending order
    '''
    def search_for_comments_by_post_id(self, post_id, list_of_comments):
        try:
            res_comments = requests.get(f'https://api.pushshift.io/reddit/search/comment/?link_id={post_id}&size=100&sort_type=score&sort=desc').json()
        except e:
            print("An error occurred")
            print(e)
            print(res_comments)

        for comment in res_comments['data']:
            # this seems to filter a lot of the comments, when applying this it literally changed from 281 to 500
            if comment['body'] != '[removed]' and comment['body'] != '[deleted]':
                list_of_comments.append(comment['body'])

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



if __name__ == "__main__":
    r = RedditAPI()
    format = "%m/%d/%Y %H:%M:%S"
    now = datetime.now()
    start = now - timedelta(days=100)

    end_ep = int(time.mktime(time.strptime(f'{now.month}/{now.day}/{now.year} {now.hour}:{now.minute}:{now.second}', format)))
    start_ep = int(time.mktime(time.strptime(f'{start.month}/{start.day}/{start.year} {start.hour}:{start.minute}:{start.second}', format)))

    # print(start.strftime('%s'))
    print(end_ep)
    print(start_ep)
    start_time = time.time()
    comments_list = r.search_for_posts(f'GME', 5, start_ep, end_ep, f'wallstreetbets')
    end_time = time.time() - start_time

    print('Comments List: ', comments_list)
    print(f'Length of comment list: {len(comments_list)}')
    print(f'\n\nTIME ELAPSED: {end_time}s\n\n')