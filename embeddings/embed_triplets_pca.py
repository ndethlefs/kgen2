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

# here the PCA part...

fig = plt.figure(0, figsize=(6, 5))
#fig = plt.figure(0, figsize=(4.8, 4))
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

'''
pca = PCA(n_components=3)
X_reduced = pca.fit_transform(x)
print(pca.get_precision())
'''

#x = np.random.randn(100, 50)
print(x.shape)
#sys.exit(0)

pca = PCA(n_components=4)
pca.fit(x)

U, S, VT = np.linalg.svd(x - x.mean(0))
X_train_pca = pca.transform(x)
X_train_pca2 = (x - pca.mean_).dot(pca.components_.T)
assert_array_almost_equal(X_train_pca, X_train_pca2)
X_projected = pca.inverse_transform(X_train_pca)
X_projected2 = X_train_pca.dot(pca.components_) + pca.mean_
assert_array_almost_equal(X_projected, X_projected2)

print('pca_components',pca.components_)
print('x[0]',x[0])
print('X_projected[0]',X_projected[0])

loss = ((x - X_projected) ** 2).mean()

print('loss=', loss)
print(pca.labels_)



print("--- %s seconds ---" % (time.time() - start_time))





