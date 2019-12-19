from __future__ import unicode_literals
# Import Dataset
import pandas as pd
import gensim
import sys
import nltk
from gensim.utils import simple_preprocess
import spacy
import tweepy
import nltk
# nltk.download('stopwords')
from nltk.stem.porter import *
import gensim.corpora as corpora
import datetime
import pprint
reload(sys)
sys.setdefaultencoding('utf8')
import preprocessor as p
# ==================================== #
pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)

# Get Data FROM Twitter
consumer_key = 'dZFi155G5IDOhM47hxYJBmcjb'
consumer_secret = 'O49HwYzLVPIPfp2nSVsPmhu7MUulE5x2bUKDjiMctt1JlqN4LG'
access_token = '1178195198389583872-mHeRnLHYGxAprLgeBrMLn3CcT4ny7P'
access_token_secret = 'BWpvwlOcafaLUjygWmaYquzXjBFGJ56AUImA7XQlUTG9L'

# OAuth process, using the keys and tokens
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

# Creation of endDate
startDate = datetime.datetime(2019, 12, 31, 0, 0, 0)
endDate = datetime.datetime(2019, 12, 1, 0, 0, 0)
# Creation of the actual interface, using authentication
api = tweepy.API(auth)
tweets = pd.DataFrame(columns=['tweet_text'])
for status in tweepy.Cursor(api.user_timeline, screen_name='@elonmusk', count=200, tweet_mode="extended").items():
    if status.created_at >= endDate:
        if status.created_at <= startDate:
            tweet = status.full_text
            tweets = tweets.append({
                'tweet_text': tweet,
                'created_date': status.created_at
            }, ignore_index=True)
    else:
        break

print(tweets)
# ==================================== #
data = list(tweets['tweet_text'])

def sent_to_words(sentences):
    for sentence in sentences:
        yield gensim.utils.simple_preprocess(str(p.clean(sentence)), deacc=True)  # deacc=True removes punctuations


data_words = list(sent_to_words(data))

print(data_words)

# ==================================== #
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)  # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)

# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)


# See trigram example
# print(trigram_mod[bigram_mod[data_words[0]]])
# ===================================== #

# Define functions for stopwords, bigrams, trigrams and lemmatization


def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]


def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]


def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]


def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


# # NLTK Stop words

from nltk.corpus import stopwords

stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

# Remove Stop Words
data_words_nostops = remove_stopwords(data_words)

# Form Bigrams
data_words_bigrams = make_bigrams(data_words_nostops)

# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
# python3 -m spacy download en
nlp = spacy.load('en', disable=['parser', 'ner'])

# Do lemmatization keeping only noun, adj, vb, adv
data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

# print(data_lemmatized[:1])
# ====================================== #
# Create Dictionary
id2word = corpora.Dictionary(data_lemmatized)

# Create Corpus
texts = data_lemmatized

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

# View
# print(corpus[:1])

# =================================== #
# Build LDA model
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                            id2word=id2word,
                                            num_topics=20,
                                            random_state=100,
                                            update_every=1,
                                            chunksize=100,
                                            passes=5,
                                            alpha='auto',
                                            per_word_topics=True)


# Print the Keyword in the 10 topics
# pprint(lda_model.print_topics())
# doc_lda = lda_model[corpus]
