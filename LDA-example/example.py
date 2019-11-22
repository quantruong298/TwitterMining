# coding=utf-8
import requests
from requests_oauthlib import OAuth1
import sys
import pandas as pd
import nltk
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
import tweepy

reload(sys)
sys.setdefaultencoding('utf8')

np.random.seed(2018)
nltk.download('wordnet')


def lemmatize_stemming(text):
    return SnowballStemmer('english').stem(WordNetLemmatizer().lemmatize(text, pos='v'))


def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result


# Get Data FROM Twitter
consumer_key = 'dZFi155G5IDOhM47hxYJBmcjb'
consumer_secret = 'O49HwYzLVPIPfp2nSVsPmhu7MUulE5x2bUKDjiMctt1JlqN4LG'
access_token = '1178195198389583872-mHeRnLHYGxAprLgeBrMLn3CcT4ny7P'
access_token_secret = 'BWpvwlOcafaLUjygWmaYquzXjBFGJ56AUImA7XQlUTG9L'

# OAuth process, using the keys and tokens
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

# Creation of the actual interface, using authentication
api = tweepy.API(auth)
data = pd.DataFrame(columns=['tweet_text'])
for status in tweepy.Cursor(api.user_timeline, screen_name='@realDonaldTrump', count=200,
                            tweet_mode="extended").items():
    data = data.append({'tweet_text': re.sub(r"http\S+", "", status.full_text)}, ignore_index=True)

data_text = data[['tweet_text']]
data_text['index'] = data_text.index
documents = data_text

# print(len(documents))
# print(documents[:5])

# doc_sample = documents[documents['index'] == 10].values[0][0]
# # print('original document: ')
# words = []
# for word in doc_sample.split(' '):
#     words.append(word)
# # print(words)
# # print('\n\n tokenized and lemmatized document: ')
# print(preprocess(doc_sample))

processed_docs = documents['tweet_text'].map(preprocess)

dictionary = gensim.corpora.Dictionary(processed_docs)

count = 0
for k, v in dictionary.iteritems():
    print(k, v)
    count += 1
    if count > 10:
        break

# dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)

bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
print (bow_corpus[10])

# bow_doc_4310 = bow_corpus[10]
# for i in range(len(bow_doc_4310)):
#     print("Word {} (\"{}\") appears {} time.".format(bow_doc_4310[i][0],
#                                            dictionary[bow_doc_4310[i][0]],
#                                                      bow_doc_4310[i][1]))

lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=10, id2word=dictionary, passes=2, workers=2)

for idx, topic in lda_model.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))

# for index, score in sorted(lda_model[bow_corpus[4310]], key=lambda tup: -1*tup[1]):
#     print("\nScore: {}\t \nTopic: {}".format(score, lda_model.print_topic(index, 10)))


# unseen_document = "Expanding the diversity of therapies in the pipeline and more efficiently recruiting patients for clinical trials will increase our odds of discovering a breakthrough. I’m hopeful it will lead to an intervention that reduces the impact of Alzheimer"
# bow_vector = dictionary.doc2bow(preprocess(unseen_document))
# for index, score in sorted(lda_model[bow_vector], key=lambda tup: -1*tup[1]):
#     print("Score: {}\t Topic: {}".format(score, lda_model.print_topic(index, 5)))
