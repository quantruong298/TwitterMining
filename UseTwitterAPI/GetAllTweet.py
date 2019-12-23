from __future__ import unicode_literals
import tweepy
import sys
import gensim
from gensim.utils import simple_preprocess
import pandas
import spacy
import re
import json
from flask_api import FlaskAPI
from flask_cors import CORS
from flask import request
from datetime import datetime
from nltk.stem.porter import *
import gensim.corpora as corpora
import preprocessor
reload(sys)
sys.setdefaultencoding('utf8')
# Setting for display full data in DataFrame
# pandas.set_option('display.max_rows', None)
# pandas.set_option('display.max_columns', None)
# pandas.set_option('display.width', None)
# pandas.set_option('display.max_colwidth', -1)

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

app = FlaskAPI(__name__)
CORS(app)

# Create global variables
tweet_list = pandas.DataFrame(columns=['created_at', 'tweet_text'])
topics = ""
bigram_mod = []
trigram_mod = []
# # NLTK Stop words
from nltk.corpus import stopwords

stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])
# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
# python3 -m spacy download en
nlp = spacy.load('en', disable=['parser', 'ner'])


# ======================= #


def sent_to_words(sentences):
    for sentence in sentences:
        yield gensim.utils.simple_preprocess(str(preprocessor.clean(sentence)), deacc=True)  # deacc=True removes punctuations


# Define functions for stopwords, bigrams, trigrams and lemmatization


def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]


def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]


def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]


def lemmatization(texts, allowed_postags=['NOUN']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


# ======================= #


@app.route('/', methods=['POST'])
def getTweets():
    data = request.get_json()
    screenname = data['screenName']
    if data['startDate'] == '':
        startDate = datetime.today()
    else:
        startDate = datetime.strptime(data['startDate'], '%Y-%m-%d')

    global tweet_list
    tweet_list = tweet_list[0:0]
    if data['endDate'] != '':
        endDate = datetime.strptime(data['endDate'], '%Y-%m-%d')
        for status in tweepy.Cursor(api.user_timeline, screen_name=screenname, count=200,
                                    tweet_mode="extended").items():
            if status.created_at >= endDate:
                if status.created_at <= startDate:
                    tweet_list = tweet_list.append({
                        'created_at': datetime.strftime(status.created_at, '%m-%d-%Y'),
                        'tweet_text': re.sub(r"http\S+", "", status.full_text)
                    }, ignore_index=True)
            else:
                break
    else:
        for status in tweepy.Cursor(api.user_timeline, screen_name=screenname, count=200,
                                    tweet_mode="extended").items():
            if status.created_at <= startDate:
                tweet_list = tweet_list.append({
                    'created_at': datetime.strftime(status.created_at, '%m-%d-%Y'),
                    'tweet_text': re.sub(r"http\S+", "", status.full_text)
                }, ignore_index=True)

    response = tweet_list.to_json(orient='records')
    return response


@app.route('/topics/<num>')
def findTopics(num):
    # Declare globals
    global bigram_mod
    global trigram_mod
    # Make a list of tweet_text
    data = list(tweet_list['tweet_text'])

    # Make a simple_processing
    data_words = list(sent_to_words(data))

    # Make bigram and trigram
    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)  # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[data_words], threshold=100)

    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)

    # Remove Stop Words
    data_words_nostops = remove_stopwords(data_words)

    # Form Bigrams
    data_words_bigrams = make_bigrams(data_words_nostops)

    # Do lemmatization keeping only noun, adj, vb, adv
    data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN'])

    # Create Dictionary
    id2word = corpora.Dictionary(data_lemmatized)

    # Create Corpus
    texts = data_lemmatized

    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]

    # Build LDA model
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                id2word=id2word,
                                                num_topics=num,
                                                random_state=100,
                                                update_every=1,
                                                chunksize=100,
                                                passes=100,
                                                alpha='auto',
                                                per_word_topics=True)

    response = json.dumps(lda_model.print_topics())
    return response


if __name__ == '__main__':
    app.run()
