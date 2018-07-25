import os
print(os.getcwd())
import word2vec
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.plotly as py 
import plotly.tools as tls 
from plotly.graph_objs import *

#print "training to group similar words into phrase"

#word2vec.word2phrase('/Users/nina/Downloads/refs.txt', '/Users/nina/Downloads/ref-phrases', verbose=True)

#print "training model for words..."

#word2vec.word2vec('wiki_vocab.txt', 'wiki_vocab-100.bin', size=100, verbose=True)


#print "training clusters"
#word2vec.word2clusters('wiki_vocab.txt', 'vocab-clusters-100.txt', 100, verbose=True)

print("loading model")

#model = word2vec.load('text8-100.bin')
model = word2vec.load('wiki_vocab-100.bin')

#print "vocabulary..."

#print model.vocab

#print('loading clusters...')
#clusters = word2vec.load_clusters('vocab-clusters-100.txt')
weights = []

word1 = 'football'
word2 = 'hockey'
word3 = 'politician'
word4 = 'actor'
word5 = 'indian'

#word_list = ['computer', 'viper', 'science', 'digital', 'cpu', 'gpu']
word_list = [word1, word2, word3, word4, word5]
word_list1 = [word1, word2, word3, word4, word5]
word_list2 = []

for w1, w1tem in enumerate(word_list):
	indexes, metrics = model.cosine(w1tem)
	responses = model.generate_response(indexes, metrics)
	for r in responses:
		m = r[0].replace("u'", '').replace("'", "")
		word_list1.append(m)
		word_list2.append((str(m), w1))

print(word_list2)

file1 = open('t8-vectors1.txt', 'w')
file2 = open('t8-labels1.txt', 'w')
file3 = open('words.txt', 'w')

label_names = {}
label_count = 0

for w, wtem in enumerate(word_list2):
	file3.writelines(str(wtem[0])+'\n')
	vector = []
	wv = model[wtem[0]]
	for word in wv:
		vector.append(word)
	weights.append(vector)
	file1.writelines(str(vector).replace(']', '').replace('[', '').replace(',', '	') + '\n')	

	file2.writelines(str(wtem[1]) + '\n')

file1.close()	
file2.close()	
file3.close()

print(word1)
indexes, metrics = model.cosine(word1)
print(model.generate_response(indexes, metrics))

print(word2)
indexes, metrics = model.cosine(word2)
print(model.generate_response(indexes, metrics))

print(word3)
indexes, metrics = model.cosine(word3)
print(model.generate_response(indexes, metrics))

print(word4)
indexes, metrics = model.cosine(word4)
print(model.generate_response(indexes, metrics))

print(word5)
indexes, metrics = model.cosine(word5)
print(model.generate_response(indexes, metrics))


#print model['sunny'][:10]

'''
print 'vocab length', len(model.vocab)

print "shape of vectors..."

print model.vectors.shape

print "printing vectors"

print model.vectors

print "shape of ball entry"

print model['car'].shape

print model['lorry']

print "***10th index of entry red"

print model['airplane'][:10]

print "***10th index of entry blue"

print model['digger'][:10]

print "***10th index of entry big"

print model['restaurant'][:10]

print "getting similar words to country"

indexes, metrics = model.cosine('science')
print indexes, metrics

print "print the actual words"
print model.vocab[indexes]

print "actual similar words with probs"

print model.generate_response(indexes, metrics)

print "showing them as a list"

print model.generate_response(indexes, metrics).tolist()
'''