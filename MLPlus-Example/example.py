import re
# import numpy as np
import pandas as pd
# from pprint import pprint
#
# # Gensim
# import gensim
# import gensim.corpora as corpora
# from gensim.utils import simple_preprocess
# from gensim.models import CoherenceModel
#
# # spacy for lemmatization
# import spacy
#
# # Plotting tools
# import pyLDAvis
# import pyLDAvis.gensim  # don't skip this
# import matplotlib.pyplot as plt
# # %matplotlib inline
#
# # Enable logging for gensim - optional
# import logging
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)
#
# import warnings
# warnings.filterwarnings("ignore",category=DeprecationWarning)

# ==========DONE STEP 3=====================
# NLTK Stop words
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

# ==========DONE STEP 5=====================

# Import Dataset
df = pd.read_json('https://raw.githubusercontent.com/selva86/datasets/master/newsgroups.json')
# print(df.target_names.unique())
df.target_names.unique()
df.head()

# ==========DONE STEP 6=====================
# Convert to list
data = df.content.values.tolist()

# Remove Emails
data = [re.sub('\S*@\S*\s?', '', sent) for sent in data]

# Remove new line characters
data = [re.sub('\s+', ' ', sent) for sent in data]

# Remove distracting single quotes
data = [re.sub("\'", "", sent) for sent in data]

print(data[:1])

# ==========DONE STEP 7=====================
