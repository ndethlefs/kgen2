from random import randint
from numpy import array
from numpy import argmax
from numpy import array_equal
from numpy import asarray
from numpy import zeros
from keras.models import Sequential
from keras.layers import LSTM, Embedding, GRU, Bidirectional
from attention_decoder import AttentionDecoder
from keras.preprocessing.text import one_hot
from keras.utils import to_categorical
import numpy as np
import nltk
import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import sys
from keras.callbacks import EarlyStopping, ModelCheckpoint
import time

os.environ["CUDA_VISIBLE_DEVICES"]="-1"

class CharacterTable(object):
    '''
    Given a set of characters:
    + Encode them to a one hot integer representation
    + Decode the one hot integer representation to their character output
    + Decode a vector of probabilities to their character output
    '''
    def __init__(self, chars, maxlen):
        self.chars = sorted(set(chars))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))
        self.maxlen = maxlen

    def encode(self, C, maxlen=None):
        maxlen = maxlen if maxlen else self.maxlen
        X = np.zeros((maxlen, len(self.chars)))
        for i, c in enumerate(C):
        	if c in self.char_indices:
	            X[i, self.char_indices[c]] = 1
	        else:
	         	X[i, self.char_indices['unknown']] = 1   
        return X

    def encode2D(self, C, maxlen=None):
        maxlen = maxlen if maxlen else self.maxlen
        X = np.zeros((len(chars)))
        for c in C:
        	ind = self.char_indices[c]
        	X[ind] = 1
        return X

    def decode(self, X, calc_argmax=True):
        if calc_argmax:
            X = X.argmax(axis=-1)
        return ' '.join(self.indices_char[x] for x in X)



inputs = []
outputs = []
sentence_start_token = "BOS"
sentence_end_token = "EOS"
token_string = ""
max_len = 13


path_train = "./100_clusters/all_train/"
path_test = "./100_clusters/all_test/"
path_out = "./100_clusters/out/"

print(os.listdir(path_train))

for file in os.listdir(path_train):
	f = file
	out_file = open(path_out + file, 'w')
	inputs = []
	outputs = []
	inputs_test = []
	outputs_test = []	
	token_string = ""
	MAXLEN = 0
	print('processing... ', f)
	print('reading training set...')			
	for line in open(path_train + file, 'r'):
			part0 = line.split("===")[0]
			part1 = line.split("===")[1].replace("\n", "")		
			p_0 = part0.split()
			p_1 = part1.split()			
			for p, ptem in enumerate(p_1):
				if p>20:	
					p_1.remove(ptem)		
			part0 = "%s %s %s" % (sentence_start_token, ' '.join(p_0), sentence_end_token)
			part1 = "%s %s %s" % (sentence_start_token, ' '.join(p_1), sentence_end_token)					
			
			token_string = token_string + part1 + " "	
			token_string = token_string + part0 + " "			
			inputs.append(part0.split())
			outputs.append(part1.split())		
			if len(part0.split()) > MAXLEN:
				MAXLEN = len(part0.split())
			elif len(part1.split()) > MAXLEN:
				MAXLEN = len(part1.split())	
	
	print('reading test set...')		
	for line in open(path_test + file, 'r'):
			part0 = line.split("===")[0]
			part1 = line.split("===")[1].replace("\n", "")		
			p_0 = part0.split()
			p_1 = part1.split()			
			for p, ptem in enumerate(p_1):
				if p>20:	
					p_1.remove(ptem)								
			part0 = "%s %s %s" % (sentence_start_token, ' '.join(p_0), sentence_end_token)								
			part1 = "%s %s %s" % (sentence_start_token, ' '.join(p_1), sentence_end_token)											
							
			token_string = token_string + part1 + " "
			token_string = token_string + part0 + " "			
			inputs_test.append(part0.split())
			outputs_test.append(part1.split())		
			if len(part0.split()) > MAXLEN:
				MAXLEN = len(part0.split())
			elif len(part1.split()) > MAXLEN:
				MAXLEN = len(part1.split())			
				
	for i, inp in enumerate(inputs):
		while len(inp) < MAXLEN:
			inp.append("PAD")
			inputs[i] = inp
	for o, outp in enumerate(outputs):
		while len(outp) < MAXLEN:
			outp.append("PAD")
			outputs[o] = outp

	for i, inp in enumerate(inputs_test):
		while len(inp) < MAXLEN:
			inp.append("PAD")
			inputs_test[i] = inp
	for o, outp in enumerate(outputs_test):
		while len(outp) < MAXLEN:
			outp.append("PAD")
			outputs_test[o] = outp

	token_string =" ".join(token_string.split())
	tokens = nltk.word_tokenize(token_string)
	tokens.append("PAD")
	tokens.append("BOS")
	tokens.append("EOS")
	tokens.append("unknown")	
	print('corpus length:', len(tokens), 'tokens.')
	chars = set(tokens)
	print(chars)
	ctable = CharacterTable(chars, MAXLEN)
	print('Found', len(set(tokens)), 'unique words.')
	print("MAXLEN=", MAXLEN)
	
	questions = []
	expected = []

	questions = inputs
	expected = outputs
	
	test_questions = inputs_test
	test_expected = outputs_test		
	
	print('Total number of examples:', len(questions))

	print('Vectorisation...')
	X_train = np.zeros((len(questions), MAXLEN, len(chars)), dtype=np.bool)
	y_train = np.zeros((len(questions), MAXLEN, len(chars)), dtype=np.bool)

	X_val = np.zeros((len(test_questions), MAXLEN, len(chars)), dtype=np.bool)
	y_val = np.zeros((len(test_questions), MAXLEN, len(chars)), dtype=np.bool)

	for i, sentence in enumerate(questions):
		print(i, 'x encoding')
		X_train[i] = ctable.encode(sentence, maxlen=MAXLEN)
	for i, sentence in enumerate(expected):
		print(i, 'y encoding')	
		y_train[i] = ctable.encode(sentence, maxlen=MAXLEN)

	for i, sentence in enumerate(test_questions):
		print(i, 'x test encoding')			
		X_val[i] = ctable.encode(sentence, maxlen=MAXLEN)
	for i, sentence in enumerate(test_expected):
		print(i, 'y test encoding')			
		y_val[i] = ctable.encode(sentence, maxlen=MAXLEN)
		
	print(X_train.shape)
	print(X_val.shape)
	print(y_train.shape)
	print(y_val.shape)	
	
	print("build model...")
	
	model_weights = "./weights/" + f + "_weights.h5"     

	callbacks = [EarlyStopping(monitor='val_loss', patience=2),
 	  	         ModelCheckpoint(filepath=model_weights, monitor='val_loss', save_best_only=True)]

	# define model
	model = Sequential()
	model.add(LSTM(256, input_shape=(MAXLEN, len(chars)), return_sequences=True))
	model.add(LSTM(256, return_sequences=True))
	model.add(Bidirectional(LSTM(100, return_sequences=True), input_shape=(256, 100)))
	#model.add(LSTM(50, return_sequences=True))
	model.add(AttentionDecoder(100, len(chars)))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
	model.summary()

	# train LSTM
	for epoch in range(500):
		# generate new random sequence
	#	X,y = get_pair(n_timesteps_in, n_timesteps_out, vocab_size)
		# fit model for one epoch on this sequence
		model.fit(X_train, y_train, epochs=1, verbose=2)
		model.save_weights(model_weights)	
		#score, acc = model.evaluate(X_val, y_val, batch_size=BATCH_SIZE)
		#print(score)
		
		for i in range(1):
			ind = np.random.randint(0, len(X_val))
			rowX, rowy = X_val[np.array([ind])], y_val[np.array([ind])]
			preds = model.predict_classes(rowX, verbose=0)
			q = ctable.decode(rowX[0])
			correct = ctable.decode(rowy[0])
			guess = ctable.decode(preds[0], calc_argmax=False)		
			print('q', q, '\n', 'correct', correct, '\n', 'guess', guess, '\n',)		
	
	#	model.load_weights(model_weights)	
	for i, item in enumerate(X_val):
		print("TESTING")
		rowX, rowy = X_val[np.array([i])], y_val[np.array([i])]		
		preds = model.predict_classes(rowX, verbose=0)
		q = ctable.decode(rowX[0])
		correct = ctable.decode(rowy[0])
		guess = ctable.decode(preds[0], calc_argmax=False)		
		print('q', q, '\n', 'correct', correct, '\n', 'guess', guess, '\n',)
		out_file.writelines(str(q) + ", " + str(correct) + ", " + str(guess) + "\n")
	out_file.close()	
	
	
	
	
	