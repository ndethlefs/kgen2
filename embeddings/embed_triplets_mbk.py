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

#X_embedded = TSNE(n_components=3).fit_transform(x)
X_embedded = PCA(n_components=4).fit_transform(x)

print('X_embedded',X_embedded)

# use this command to use embeddings, don't use otherwise...
x = X_embedded

mbk =  MiniBatchKMeans(init='k-means++', n_clusters=3, batch_size=2,
                      n_init=10, max_no_improvement=10, verbose=0)
mbk.fit(x) 
print('score',mbk.score)

labels = mbk.labels_

for i, item in enumerate(x):
	print(item, 'cluster=', labels[i], 'entry=', y[i])
	
	
# test prediction making... 
predicted = mbk.predict(x[0])	
print("predicted", predicted)

# can find closest cluster point and make a prediction using the trained labeller... 
print(find_closest(x[0], x, 'kl'), 'predicted for p=', mbk.predict(x[0]), "cluster for closest=", mbk.predict(find_closest(x[0], x, 'kl')[0]))
	

print("--- %s seconds ---" % (time.time() - start_time))





