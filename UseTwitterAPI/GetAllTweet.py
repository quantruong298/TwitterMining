import tweepy
import sys
import pandas
import re
# Setting for display full data in DataFrame
pandas.set_option('display.max_rows', None)
pandas.set_option('display.max_columns', None)
pandas.set_option('display.width', None)
pandas.set_option('display.max_colwidth', -1)

# Consumer keys and access tokens, used for OAuth
consumer_key = 'dZFi155G5IDOhM47hxYJBmcjb'
consumer_secret = 'O49HwYzLVPIPfp2nSVsPmhu7MUulE5x2bUKDjiMctt1JlqN4LG'
access_token = '1178195198389583872-mHeRnLHYGxAprLgeBrMLn3CcT4ny7P'
access_token_secret = 'BWpvwlOcafaLUjygWmaYquzXjBFGJ56AUImA7XQlUTG9L'

# OAuth process, using the keys and tokens
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

# Creation of the actual interface, using authentication
api = tweepy.API(auth)
tweet_list = pandas.DataFrame(columns=['tweet_text'])
for status in tweepy.Cursor(api.user_timeline, screen_name='@BillGates', count=200, tweet_mode="extended").items():
    tweet_list = tweet_list.append({'tweet_text':re.sub(r"http\S+", "", status.full_text)}, ignore_index=True)

print (tweet_list)


