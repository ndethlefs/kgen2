from random import randint
from numpy import array
from numpy import argmax
from numpy import array_equal
from numpy import asarray
from numpy import zeros
from keras.models import Sequential
from keras.layers import LSTM, Embedding, GRU
from attention_decoder import AttentionDecoder
from keras.preprocessing.text import one_hot
from keras.utils import to_categorical
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import sys
import time

# define documents
docs = ['Well done!',
		'Good work',
		'Great effort',
		'nice work',
		'Excellent!',
		'Weak',
		'Poor effort!',
		'not good',
		'poor work',
		'Could have done better.']
# define class labels
#labels = array([1,1,1,1,1,0,0,0,0,0])
#labels = array([[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1]])

# prepare tokeniser and obtain vocabulary
t = Tokenizer()
t.fit_on_texts(docs)
vocab_size = len(t.word_index) + 1
max_len = 4

# one hot encode sequence
def one_hot_encode(sequence, n_unique):
	encoding = list()
	for value in sequence:
		vector = [0 for _ in range(n_unique)]
		vector[value] = 1
		encoding.append(vector)
	return array(encoding)


# decode a one hot encoded string
def one_hot_decode(encoded_seq):
	return [argmax(vector) for vector in encoded_seq]

# get a one-hot encoding of the inputs
encoded_docs = [one_hot(d, vocab_size) for d in docs]
# pad documents to a max length of 4 words
padded_docs = pad_sequences(encoded_docs, maxlen=max_len, padding='post')
encoded_docs = np.asarray([one_hot_encode(d, vocab_size) for d in padded_docs])
X,y = encoded_docs,encoded_docs
print(X.shape)
print(y.shape)

# From hereon, prepare an alternative representation in 2D... 

# integer encode the documents
# prepare dictionaries for encoding (word_index) and for decoding (index_word)
encoded_docs = t.texts_to_sequences(docs)
word_index = t.word_index
index_word = {v:k for k, v in word_index.items()}

def decode_sequence(seq):
	decoded = ""
	for s in seq:
		if not s==0:
			decoded = decoded + index_word[int(s)] + " "
	return decoded.strip()	


# pad documents to a max length of 4 words (should be the maximum sequence length)
padded_docs = pad_sequences(encoded_docs, maxlen=max_len, padding='post')
print('padded_docs', padded_docs)

X_em = padded_docs
y_em = padded_docs
print(X_em)
print(X_em.shape)

#X = padded_docs
#y = padded_docs


# load the whole embedding into memory
embeddings_index = dict()
f = open('./glove_data/glove.6B/glove.6B.100d.txt')
for line in f:
	values = line.split()
	word = values[0]
	coefs = asarray(values[1:], dtype='float32')
	embeddings_index[word] = coefs
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))
# create a weight matrix for words in training docs
embedding_matrix = zeros((vocab_size, 100))
for word, i in t.word_index.items():
	embedding_vector = embeddings_index.get(word)
	if embedding_vector is not None:
		embedding_matrix[i] = embedding_vector

'''
e = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=4, trainable=False)
model.add(e)
model.add(GRU(100, return_sequences=True))
'''


e = Embedding(vocab_size, 150, embeddings_initializer='uniform', input_length=None)

print('X', X.shape)
print('y', y.shape)
sys.exit(0)

# define model
model = Sequential()
#model.add(e)
model.add(GRU(150, input_shape=(max_len, vocab_size), return_sequences=True))
model.add(GRU(150, return_sequences=True))
model.add(GRU(50, return_sequences=True))
model.add(AttentionDecoder(50, vocab_size))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.summary()

# train LSTM
for epoch in range(50):
	# generate new random sequence
#	X,y = get_pair(n_timesteps_in, n_timesteps_out, vocab_size)
	# fit model for one epoch on this sequence
	model.fit(X, y, epochs=1, verbose=2)

print('X', X)

# evaluate LSTM
total, correct = 10, 0
for _ in range(total):
#	X,y = get_pair(n_timesteps_in, n_timesteps_out, vocab_size)
	yhat = model.predict(X, verbose=0)
	if array_equal(one_hot_decode(y[0]), one_hot_decode(yhat[0])):
		correct += 1
print('Accuracy: %.2f%%' % (float(correct)/float(total)*100.0))

# spot check some examples
for _ in range(10):
#	X,y = get_pair(n_timesteps_in, n_timesteps_out, vocab_size)
	yhat = model.predict(X, verbose=0)
	print('Expected:', one_hot_decode(y[0]), 'Predicted', one_hot_decode(yhat[0]))
	
print("--- %s seconds ---" % (time.time() - start_time))	
	
	
	