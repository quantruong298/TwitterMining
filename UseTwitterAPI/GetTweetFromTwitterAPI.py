import requests
from requests_oauthlib import OAuth1
import json
import pandas

# Setting for display full data in DataFrame
# pandas.set_option('display.max_rows', None)
# pandas.set_option('display.max_columns', None)
# pandas.set_option('display.width', None)
# pandas.set_option('display.max_colwidth', -1)


# API Url to get all tweets in someone's Timeline
# Use '&tweet_mode=extended' to display full text of tweet
screen_name = 'BillGates'
url = 'https://api.twitter.com/1.1/statuses/user_timeline.json?screen_name=' + screen_name + '&tweet_mode=extended'


# Data for Twitter Authentication
client_key = 'dZFi155G5IDOhM47hxYJBmcjb'
client_secret = 'O49HwYzLVPIPfp2nSVsPmhu7MUulE5x2bUKDjiMctt1JlqN4LG'
resource_owner_key = '1178195198389583872-mHeRnLHYGxAprLgeBrMLn3CcT4ny7P'
resource_owner_secret = 'BWpvwlOcafaLUjygWmaYquzXjBFGJ56AUImA7XQlUTG9L'

oauth = OAuth1(client_key,
               client_secret,
               resource_owner_key,
               resource_owner_secret,
               signature_type='auth_header')
response = requests.get(url, auth=oauth)
tweets = response.json()
tweet_texts = pandas.DataFrame(data=[tweet['full_text'] for tweet in tweets], columns=['tweet_text'])
print (tweet_texts)
