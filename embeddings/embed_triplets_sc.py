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
from sklearn.cluster import MiniBatchKMeans, KMeans, spectral_clustering
from kb_utils import *

start_time = time.time()

file = "wiki-triplets-tiny.txt"

reader = TripletsReader(file)
reader.storeTriplets(file)
graph_delex = reader.delexicalise_graph()
sorted_data = sort_data(graph_delex)
x, y = get_x_y(sorted_data)
max_len = get_max_len(x, y)

# prepare tokeniser and obtain vocabulary
t = Tokenizer()
t.fit_on_texts(x)
vocab_size = len(t.word_index) + 1

# integer encode the documents
# prepare dictionaries for encoding (word_index) and for decoding (index_word)
encoded_docs = t.texts_to_sequences(x)
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

x = padded_docs
print(x)

# here the k Means part...


sc =  spectral_clustering(x, n_clusters=2, eigen_solver='arpack')
sc.fit(x) 
print('score',sc.score)

labels = sc.labels_

for i, item in enumerate(x):
	print(item, 'cluster=', labels[i], 'entry=', y[i])


#>>> print(k_means.labels_[::10])
#[1 1 1 1 1 0 0 0 0 0 2 2 2 2 2]




print("--- %s seconds ---" % (time.time() - start_time))





