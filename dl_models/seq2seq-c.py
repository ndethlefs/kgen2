# -*- coding: utf-8 -*-
'''An implementation of sequence to sequence learning for performing addition
Input: "535+61"
Output: "596"
Padding is handled by using a repeated sentinel character (space)

Input may optionally be inverted, shown to increase performance in many tasks in:
"Learning to Execute"
http://arxiv.org/abs/1410.4615
and
"Sequence to Sequence Learning with Neural Networks"
http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf
Theoretically it introduces shorter term dependencies between source and target.

Two digits inverted:
+ One layer LSTM (128 HN), 5k training examples = 99% train/test accuracy in 55 epochs

Three digits inverted:
+ One layer LSTM (128 HN), 50k training examples = 99% train/test accuracy in 100 epochs

Four digits inverted:
+ One layer LSTM (128 HN), 400k training examples = 99% train/test accuracy in 20 epochs

Five digits inverted:
+ One layer LSTM (128 HN), 550k training examples = 99% train/test accuracy in 30 epochs

'''

from __future__ import print_function
from keras.models import Sequential
#from keras.engine.training import slice_X
from keras.layers import Activation, TimeDistributed, Dense, RepeatVector, recurrent, Embedding, Flatten, Reshape
import numpy as np
from six.moves import range
import nltk
from keras.preprocessing.sequence import pad_sequences
#from bleu_simpler import *

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
            X[i, self.char_indices[c]] = 1
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


class colors:
    ok = '\033[92m'
    fail = '\033[91m'
    close = '\033[0m'
    highlight = '\033[94m'    

# Parameters for the model and dataset
TRAINING_SIZE = 50000
#DIGITS = 3
INVERT = True
# Try replacing GRU, or SimpleRNN
RNN = recurrent.LSTM
HIDDEN_SIZE = 50
EMBED_SIZE = 50
BATCH_SIZE = 128
LAYERS = 4
EPOCHS = 5000
MAXLEN = 0 # updated when reading dataset
#INPUT_DATA_FILE = "./../data/full-datasets/refs-sem-clustered-text8.txt"
#INPUT_DATA_FILE = "./../data/full-datasets/refs-sem-clustered.txt"
#INPUT_DATA_FILE = "./../data/full-datasets/refs-sem-clustered-stopwords.txt"
#INPUT_DATA_FILE = "./../data/full-datasets/sfx-restaurants-clustered.txt"
#INPUT_DATA_FILE = "./../data/AMR/sfx-restaurants-amr-delex.txt"
#INPUT_DATA_FILE = "./../data/full-datasets/sfx-restaurants.txt"
#INPUT_DATA_FILE = "./../data/full-datasets/refs-sem.txt"
INPUT_DATA_FILE = 'train_wiki_lstm.txt'
OUTPUT_DATA_FILE = 'test_wiki_lstm.txt'
OUTPUT_FILE= INPUT_DATA_FILE.split('.txt')[0] + '-weights.h5'

# get datasets, inputs and expected outputs
text = open(INPUT_DATA_FILE).read()
print('corpus length:', len(text), 'characters.')
chars = set(text)

inputs = []
outputs = []
inputs_test = []
outputs_test = []
token_string = ""
sentence_start_token = "BOS"
sentence_end_token = "EOS"

for line in open(INPUT_DATA_FILE, 'r'):
	part0 = line.split("===")[0]
	part0 = "%s %s %s" % (sentence_start_token, part0, sentence_end_token)
	token_string = token_string + part0 + " "
	part0 = nltk.word_tokenize(part0)
	part1 = line.split("===")[1].replace("\n", "")
	part1 = "%s %s %s" % (sentence_start_token, part1, sentence_end_token)	
	token_string = token_string + part1 + " "	
	part1 = nltk.word_tokenize(part1)	
	inputs.append(part0)
	outputs.append(part1)	
	if len(part0) > MAXLEN:
		MAXLEN = len(part0)
	elif len(part1) > MAXLEN:
		MAXLEN = len(part1)
		
for line in open(OUTPUT_DATA_FILE, 'r'):
	part0 = line.split("===")[0]
	part0 = "%s %s %s" % (sentence_start_token, part0, sentence_end_token)
	token_string = token_string + part0 + " "
	part0 = nltk.word_tokenize(part0)
	part1 = line.split("===")[1].replace("\n", "")
	part1 = "%s %s %s" % (sentence_start_token, part1, sentence_end_token)	
	token_string = token_string + part1 + " "	
	part1 = nltk.word_tokenize(part1)	
	inputs_test.append(part0)
	outputs_test.append(part1)	
	if len(part0) > MAXLEN:
		MAXLEN = len(part0)
	elif len(part1) > MAXLEN:
		MAXLEN = len(part1)		

#print(inputs)

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


#chars = '0123456789+ '
token_string =" ".join(token_string.split())
tokens = nltk.word_tokenize(token_string)
tokens.append("PAD")
tokens.append("BOS")
tokens.append("EOS")
print('corpus length:', len(tokens), 'tokens.')
chars = set(tokens)
print(chars)
ctable = CharacterTable(chars, MAXLEN)
print('Found', len(set(tokens)), 'unique words.')
print("MAXLEN=", MAXLEN)


questions = []
expected = []

# To classify semantics from words:
#questions = inputs
#expected = outputs

# To classify words from semantics:
questions = outputs
expected = inputs

test_questions = outputs_test
test_expected = inputs_test

print('Total number of examples:', len(questions))

print('Vectorisation...')
X_train = np.zeros((len(questions), MAXLEN, len(chars)), dtype=np.bool)
y_train = np.zeros((len(questions), MAXLEN, len(chars)), dtype=np.bool)

X_val = np.zeros((len(test_questions), MAXLEN, len(chars)), dtype=np.bool)
y_val = np.zeros((len(test_questions), MAXLEN, len(chars)), dtype=np.bool)
#X_train1 = []
#y_train1 = []

for i, sentence in enumerate(questions):
    X_train[i] = ctable.encode(sentence, maxlen=MAXLEN)
for i, sentence in enumerate(expected):
    y_train[i] = ctable.encode(sentence, maxlen=MAXLEN)


for i, sentence in enumerate(test_questions):
    X_val[i] = ctable.encode(sentence, maxlen=MAXLEN)
for i, sentence in enumerate(test_expected):
    y_val[i] = ctable.encode(sentence, maxlen=MAXLEN)

print(X_train.shape)
print(X_val.shape)
print(y_train.shape)
print(y_val.shape)


# Shuffle (X, y) in unison as the later parts of X will almost all be larger digits
#indices = np.arange(len(y))
#np.random.shuffle(indices)
#X = X[indices]
#y = y[indices]


print('Build model...')


### Model 2 without embeddings...
model = Sequential()
# "Encode" the input sequence using an RNN, producing an output of HIDDEN_SIZE
# note: in a situation where your input sequences have a variable length,
# use input_shape=(None, nb_feature).
model.add(RNN(HIDDEN_SIZE, input_shape=(MAXLEN, len(chars))))
# For the decoder's input, we repeat the encoded input for each time step
model.add(RepeatVector(MAXLEN))
# The decoder RNN could be multiple layers stacked or a single layer
for _ in range(LAYERS):
    model.add(RNN(HIDDEN_SIZE, return_sequences=True))
# For each of step of the output sequence, decide which character should be chosen
model.add(TimeDistributed(Dense(len(chars))))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
              
#plot(model, to_file='./../plots/model-chars.png', show_shapes=True)              

#semPatterns = getSemPatterns(INPUT_DATA_FILE)
#print(semPatterns)

#model.load_weights("./data/gre4joint-delex-weights.h5")

# Train the model each generation and show predictions against the validation dataset
for iteration in range(1, EPOCHS):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(X_train, y_train, batch_size=BATCH_SIZE, nb_epoch=1,
              validation_data=(X_val, y_val))
    ###
    # Select 10 samples from the validation set at random so we can visualize errors
    for i in range(1):
        ind = np.random.randint(0, len(X_val))
        rowX, rowy = X_val[np.array([ind])], y_val[np.array([ind])]
        preds = model.predict_classes(rowX, verbose=0)
        q = ctable.decode(rowX[0])
        correct = ctable.decode(rowy[0])
        guess = ctable.decode(preds[0], calc_argmax=False)
        print('Q', colors.highlight, q.split(" PAD")[0], colors.close)        
#        print('Q', q[::-1] if INVERT else q)        
        print('T', colors.highlight, correct.split(" PAD")[0], colors.close)
        print(colors.ok + '☑ ' + colors.close if correct == guess else colors.fail + '☒ ' + colors.close, guess)
        print('---')       
        guess_bleu = guess.split(" PAD")[0].split()
#        ref_bleu = correct.split(" PAD")[0].split()
#        rl = semPatterns[q.split(" PAD")[0]]
 #       ref_list = []
  #      for x in rl:
   #     	x = " ".join(x.split())
    #    	ref_list.append(x.split())
        # change int and weights to compute BLEU-3 or BLEU-4 scores... 
   #     print("BLEU", 3, "score:", getBleu(guess_bleu, [ref_bleu], [0.25, 0.25, 0.25]))
    #    print("BLEU", 4, "score:", getBleu(guess_bleu, [ref_bleu], [0.25, 0.25, 0.25, 0.25]))        
        # use this instead when evaluating word sequence outputs...
        import nltk
        BLEUscore = nltk.translate.bleu_score.sentence_bleu([guess_bleu], correct)
        print('bleu-score', BLEUscore)        
        
#        print(semPatterns[q.split(" PAD")[0]])
 #       print("BLEU", 4, "score:", getBleu(guess_bleu, ref_list, [0.25, 0.25, 0.25, 0.25]))
  #      print("BLEU", 3, "score:", getBleu(guess_bleu, ref_list, [0.25, 0.25, 0.25]))
        print('---')
    json_string = model.to_json()
    model.save_weights(OUTPUT_FILE, overwrite=True)

#json_string = model.to_json()
#model.save_weights('my_model_weights.h5')





