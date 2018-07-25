import matplotlib.pyplot as plt
# Though the following import is not directly being used, it is required
# for 3D projection to work
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from TripletsReader import *
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from numpy.testing import assert_array_almost_equal
from sklearn.model_selection import RandomizedSearchCV
from sklearn import cluster
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from kb_utils import *
#from bleu import *
from rouge import Rouge
from random import randint
from numpy import array
from numpy import argmax
from numpy import array_equal
from numpy import asarray
from numpy import zeros
from keras.models import Sequential
from keras.layers import LSTM, Embedding, GRU, RepeatVector, TimeDistributed, Dense, Activation
from keras.preprocessing.text import one_hot
from keras.utils import to_categorical


start_time = time.time()

# resources for training

file = "wiki-triplets.txt"
reader = TripletsReader(file)
reader.load_targets('targets.json')
reader.load_d_graph('d_graph.json')
graph_delex = reader.graph_delex

# resources for testing

test_file = "wiki-triplets-test.txt"
test_reader = TripletsReader(file)
test_reader.load_targets('targets-test.json')
test_reader.load_d_graph('d_graph-test.json')
test_graph_delex = test_reader.graph_delex

vocab = {}
x = []
y = [] 
test_x = []
test_y = [] 
max_len = 0

sorted_data = {}
for key in graph_delex:
	sorted_data[key] = sorted(graph_delex[key])

for s in sorted_data:
	if s in reader.targets:
		x_s = "BOS " + " ".join(sorted_data[s]) + " EOS"
		x.append(x_s) # x is a sorted sequence of semantic attributes
		y_s = "BOS " + reader.targets[s] + " EOS"		
		y.append(y_s) # y is a delexicalised sentence describing (some of) the attributes
		if len(sorted_data[s])+2 > max_len:
			max_len = len(sorted_data[s])+2	
		if len(reader.targets[s].split())+2 > max_len:
			max_len = len(reader.targets[s].split())+2 


# do the same for test data...

test_sorted_data = {}
for key in test_graph_delex:
	test_sorted_data[key] = sorted(test_graph_delex[key])


for s in test_sorted_data:
	if s in test_reader.targets:
		x_s = "BOS " + " ".join(test_sorted_data[s]) + " EOS"
		test_x.append(x_s) # x is a sorted sequence of semantic attributes
		y_s = "BOS " + test_reader.targets[s] + " EOS"
		test_y.append(y_s) # y is a delexicalised sentence describing (some of) the attributes
		if len(test_sorted_data[s])+2 > max_len:
			max_len = len(test_sorted_data[s])+2	
		if len(test_reader.targets[s].split())+2 > max_len:
			max_len = len(test_reader.targets[s].split())+2


train_file = open('train_wiki_lstm.txt', 'w')
for i, item in enumerate(x):
	train_file.writelines(str(item) + "===" + str(y[i]) + "\n")
	print(str(item) + "===" + str(y[i]) + "\n")
train_file.close()

test_file = open('test_wiki_lstm.txt', 'w')
for i, item in enumerate(test_x):
	test_file.writelines(str(item) + "===" + str(test_y[i]) + "\n")
	print(str(item) + "===" + str(test_y[i]) + "\n")
test_file.close()

sys.exit(0)

# prepare tokeniser and obtain vocabulary

t = Tokenizer()
t.fit_on_texts(x+y+test_x+test_y)
vocab_size = len(t.word_index) + 1

print('vocab_size', vocab_size)

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
	
vocab_size = 50	
	
# get a one-hot encoding of the inputs
encoded_x = [one_hot(d, vocab_size) for d in x]
encoded_y = [one_hot(d, vocab_size) for d in y]
test_encoded_x = [one_hot(d, vocab_size) for d in test_x]
test_encoded_y = [one_hot(d, vocab_size) for d in test_y]
# pad documents to a max length of 4 words
padded_x = pad_sequences(encoded_x, maxlen=max_len, padding='post')
padded_y = pad_sequences(encoded_y, maxlen=max_len, padding='post')
test_padded_x = pad_sequences(test_encoded_x, maxlen=max_len, padding='post')
test_padded_y = pad_sequences(test_encoded_y, maxlen=max_len, padding='post')

x = np.asarray([one_hot_encode(d, vocab_size) for d in padded_x])
y = np.asarray([one_hot_encode(d, vocab_size) for d in padded_y])
test_x = np.asarray([one_hot_encode(d, vocab_size) for d in test_padded_x])
test_y = np.asarray([one_hot_encode(d, vocab_size) for d in test_padded_y])


print('Build model...')

model = Sequential()
model.add(LSTM(200, input_shape=(max_len, vocab_size)))
model.add(RepeatVector(max_len))
for _ in range(3):
    model.add(LSTM(100, return_sequences=True))
model.add(TimeDistributed(Dense(vocab_size)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


# Train the model each generation and show predictions against the validation dataset
for iteration in range(1, 10000):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(x, y, batch_size=8, nb_epoch=1,
              validation_data=(test_x, test_y))
    ###
    # Select 10 samples from the validation set at random so we can visualize errors
    for i in range(3):
        ind = np.random.randint(0, len(test_x))
        rowX, rowy = test_x[np.array([ind])], test_y[np.array([ind])]
        preds = model.predict_classes(rowX, verbose=0)
        q = one_hot_decode(rowX[0])
        print('rowX[0]',rowX[0])
        print('rowy[0]',rowy[0])        
        print('preds[0]',preds[0])                
        correct = one_hot_decode(rowy[0])
        guess = one_hot_decode(preds[0])
        correct = str(correct).replace(", ", " ").replace("[", "").replace("]", "")
        correct = str(correct.split(" 0 ")[0]).split()
        guess = str(guess).replace(", ", " ").replace("[", "").replace("]", "")   
        guess = str(guess.split(" 0 ")[0])[0:len(correct)].split()
        print('X', q)
        print('predicted=', guess)        
        print('actual=', correct)                
        print('---')
        import nltk
        BLEUscore = nltk.translate.bleu_score.sentence_bleu([guess], correct)
        print('bleu-score', BLEUscore)        
        
        rouge = Rouge()
        scores = rouge.get_scores(guess, correct)
        print('ROUGE', scores)
        print('---')
#    model.save_weights(OUTPUT_FILE, overwrite=True)




print("--- %s seconds ---" % (time.time() - start_time))
sys.exit(0)




