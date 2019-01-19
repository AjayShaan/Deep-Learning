import numpy as np
import pandas as pd
# %matplotlib inline

'''
from google.colab import files
import uuid
from google.colab import auth


def fileUploadToGCS(bucket_name='colab-sample-bucket-7491b53e-49be-11e8-b0d5-0242ac110002'):
  auth.authenticate_user()
  project_id = 'api-project-834845844624'
  !gcloud config set project {project_id}

  # Make a unique bucket to which we'll upload the file.
  # (GCS buckets are part of a single global namespace.)
  #bucket_name = 'colab-sample-bucket-' + str(uuid.uuid1())
  #print bucket_name

  !gsutil cp wiki-news-300d-1M.vec gs://{bucket_name}/wiki-news-300d-1M.vec
    
def fileDownloadFromGCS():
  auth.authenticate_user()
  project_id = 'api-project-834845844624'
  !gcloud config set project {project_id}
  !gsutil cp gs://"colab-sample-bucket-7491b53e-49be-11e8-b0d5-0242ac110002"/lyrics.csv.zip ./lyrics.csv.zip
  !gsutil cp gs://"colab-sample-bucket-7491b53e-49be-11e8-b0d5-0242ac110002"/wiki-news-300d-1M.vec ./wiki-news-300d-1M.vec
  !gsutil cp gs://"colab-sample-bucket-7491b53e-49be-11e8-b0d5-0242ac110002"/GoogleNews-vectors-negative300-SLIM.bin ./GoogleNews-vectors-negative300-SLIM.bin

#fileUploadToGCS()
fileDownloadFromGCS()
!unzip lyrics.csv.zip

# import urllib
# url = "https://github.com/eyaler/word2vec-slim/blob/master/GoogleNews-vectors-negative300-SLIM.bin.gz"
# urllib.request.urlretrieve(url,"GoogleNews-vectors-negative300-SLIM.bin.gz")
# !gunzip GoogleNews-vectors-negative300-SLIM.bin.gz

!ls -lah

import numpy as np
import pandas as pd
# %matplotlib inline
'''


df = pd.read_csv('lyrics.csv')
df.shape

df.dropna(inplace=True)

top_100_artists = df['artist'].value_counts()[:100].index

df = df[df['artist'].isin(top_100_artists)]

df['artist'].unique()

def lower(x):
    try:
        return " ".join(x.lower() for x in x.split())
    except:
        return

df['lyrics'] = df['lyrics'].apply(lower)

df['lyrics'] = df['lyrics'].str.replace('[^\w\s]','')

df.head()

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop = stopwords.words('english')

def remove_stopwords(x):
    try:
        return " ".join(x for x in x.split() if x not in stop)
    except:
        return

df['lyrics'] = df['lyrics'].apply(remove_stopwords)

df.head()

#Removing frequent words

freq = pd.Series(' '.join(df['lyrics']).split()).value_counts()[:10]

freq

freq = list(freq.index)

freq.remove('love')
freq.append('verse')
freq.append('Verse')
freq.append('chorus')

freq.append('Chorus')
freq

df['lyrics'] = df['lyrics'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))

df.head()

#Removing rare words

freq = pd.Series(' '.join(df['lyrics']).split()).value_counts()

rare_words = freq[::-1][freq[::-1] == 1].index

df['lyrics'] = df['lyrics'].apply(lambda x: " ".join(x for x in x.split() if x not in rare_words))

df.head()


df = df[['artist','lyrics']]

from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble

import pandas, xgboost, numpy, string
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers

# split the dataset into training and validation datasets 
train_x, valid_x, train_y, valid_y = model_selection.train_test_split(df['lyrics'], df['artist'])

# label encode the target variable 
encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
valid_y = encoder.fit_transform(valid_y)


#Count vectorizer

count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
count_vect.fit(df['lyrics'])

# transform the training and validation data using count vectorizer object
xtrain_count =  count_vect.transform(train_x)
xvalid_count =  count_vect.transform(valid_x)


#TF-IDF vectorizers

# word level tf-idf
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
tfidf_vect.fit(df['lyrics'])
xtrain_tfidf =  tfidf_vect.transform(train_x)
xvalid_tfidf =  tfidf_vect.transform(valid_x)

# ngram level tf-idf 
tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,2), max_features=5000)
tfidf_vect_ngram.fit(df['lyrics'])
xtrain_tfidf_ngram =  tfidf_vect_ngram.transform(train_x)
xvalid_tfidf_ngram =  tfidf_vect_ngram.transform(valid_x)


#FastText vectorizer

# load the pre-trained word-embedding vectors 
embeddings_index = {}
for i, line in enumerate(open('wiki-news-300d-1M.vec', encoding='utf8')):
    values = line.split()
    embeddings_index[values[0]] = numpy.asarray(values[1:], dtype='float32')

# create a tokenizer 
token = text.Tokenizer()
token.fit_on_texts(df['lyrics'])
word_index = token.word_index

# convert text to sequence of tokens and pad them to ensure equal length vectors 
train_seq_x = sequence.pad_sequences(token.texts_to_sequences(train_x), maxlen=30)
valid_seq_x = sequence.pad_sequences(token.texts_to_sequences(valid_x), maxlen=30)

# create token-embedding mapping
embedding_matrix = numpy.zeros((len(word_index) + 1, 300))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

embedding_matrix.shape

train_seq_x.shape


#Word2Vec vectorizer
!pip3 install gensim
from gensim.models import KeyedVectors

# Load pretrained model (since intermediate data is not included, the model cannot be refined with additional data)
model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300-SLIM.bin', binary=True)

dog = model['dog']
print(dog.shape)
print(dog[:10])

# create token-embedding mapping
w_embedding_matrix = numpy.zeros((len(word_index) + 1, 300))
for word, i in word_index.items():
    try:
        w_embedding_vector = model[word]
        w_embedding_matrix[i] = w_embedding_vector
    except:
        continue

w_embedding_matrix.shape




def train_model(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):

    classifier.fit(feature_vector_train, label)
    
    predictions = classifier.predict(feature_vector_valid)
    
    if is_neural_net:
        predictions = predictions.argmax(axis=-1)
    accuracy = metrics.accuracy_score(predictions, valid_y)

    return accuracy

	
# Naive Bayes
accuracy = train_model(naive_bayes.MultinomialNB(alpha=1e-5), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
print("NB, WordLevel TF-IDF: ")
print('Accuracy: ', accuracy)

accuracy = train_model(naive_bayes.ComplementNB(alpha=1e-5), xtrain_tfidf, train_y, xvalid_tfidf)
print("NB, WordLevel TF-IDF: ")
print('Accuracy: ', accuracy)

accuracy = train_model(naive_bayes.MultinomialNB(alpha=1e-5), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
print("NB, BigramLevel TF-IDF: ")
print('Accuracy: ', accuracy)

# RF
accuracy = train_model(ensemble.RandomForestClassifier(n_estimators=100), xtrain_tfidf, train_y, xvalid_tfidf)
print("RF, WordLevel TF-IDF: ", accuracy)


accuracy = train_model(ensemble.RandomForestClassifier(n_estimators=100), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
print("RF, WordLevel TF-IDF: ", accuracy)

# GB
accuracy = train_model(ensemble.GradientBoostingClassifier(n_estimators=100), xtrain_tfidf, train_y, xvalid_tfidf)
print("GB, WordLevel TF-IDF: ", accuracy)

# XGB
accuracy = train_model(xgboost.XGBClassifier(), xtrain_tfidf.tocsc(), train_y, xvalid_tfidf.tocsc())
print("Xgb, WordLevel TF-IDF: ", accuracy)
