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
#from bleu import *
from rouge import Rouge

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
		x.append(" ".join(sorted_data[s])) # x is a sorted sequence of semantic attributes
		y.append(reader.targets[s].split()) # y is a delexicalised sentence describing (some of) the attributes
		if len(sorted_data[s]) > max_len:
			max_len = len(sorted_data[s])	
		if len(reader.targets[s].split()) > max_len:
			max_len = len(reader.targets[s].split())		


# do the same for test data...

test_sorted_data = {}
for key in test_graph_delex:
	test_sorted_data[key] = sorted(test_graph_delex[key])

for s in test_sorted_data:
	if s in test_reader.targets:
		test_x.append(" ".join(test_sorted_data[s])) # x is a sorted sequence of semantic attributes
		test_y.append(test_reader.targets[s].split()) # y is a delexicalised sentence describing (some of) the attributes
		if len(test_sorted_data[s]) > max_len:
			max_len = len(test_sorted_data[s])	
		if len(test_reader.targets[s].split()) > max_len:
			max_len = len(test_reader.targets[s].split())


# prepare tokeniser and obtain vocabulary
t = Tokenizer()
t.fit_on_texts(x+y+test_x+test_y)
vocab_size = len(t.word_index) + 1

print('vocab_size', vocab_size)


# integer encode the documents
# prepare dictionaries for encoding (word_index) and for decoding (index_word)
encoded_docs = t.texts_to_sequences(x)
encoded_labels = t.texts_to_sequences(y)
test_encoded_docs = t.texts_to_sequences(test_x)
test_encoded_labels = t.texts_to_sequences(test_y)
word_index = t.word_index
index_word = {v:k for k, v in word_index.items()}

def decode_sequence(seq):
	decoded = ""
	for s in seq:
		if not s==0:
			decoded = decoded + index_word[int(s)] + " "
	return decoded.strip()	
	

# pad documents to a max length of z words (where z should be the maximum sequence length)
padded_docs = pad_sequences(encoded_docs, maxlen=max_len, padding='post')
padded_labels = pad_sequences(encoded_labels, maxlen=max_len, padding='post')
test_padded_docs = pad_sequences(test_encoded_docs, maxlen=max_len, padding='post')
test_padded_labels = pad_sequences(test_encoded_labels, maxlen=max_len, padding='post')

x = padded_docs
y = padded_labels
test_x = test_padded_docs
test_y = test_padded_labels


def eval_batch(x_train, y_train, x_test, y_test, classifier, components, no_clusters, dimensionality):

	cluster_finder = cluster.KMeans(n_clusters=no_clusters)
	
	if classifier=='mbk':
		cluster_finder =  MiniBatchKMeans(init='k-means++', n_clusters=no_clusters, batch_size=32, max_no_improvement=10, verbose=0)
		cluster_finder.fit(x) 
		cddd = str(cluster_finder.score)
		clll = str(cluster_finder)
		log_str = str(components) + 'score' +cddd + 'algo=' + clll + 'comp=' + str(components)  + dimensionality
		labels = cluster_finder.labels_	

	else:
		cluster_finder =  cluster.KMeans(n_clusters=no_clusters)
		cluster_finder.fit(x) 
		cddd = str(cluster_finder.score)
		clll = str(cluster_finder)
		log_str = str(components) +'score' +cddd + 'algo=' + clll + 'comp=' + str(components) + dimensionality
		labels = cluster_finder.labels_	
		
		
	clustered_x = []
	clustered_y = []		

	for c in range(0, no_clusters):
		clustered_x.append([])
		clustered_y.append([])	

	for i, item in enumerate(x_train):
		item = item.reshape(1, -1)
		predicted = cluster_finder.predict(item)	
		clustered_x[predicted[0]].append(item)
		clustered_y[predicted[0]].append(y[i])
	
	# here testing
	for q, qtem in enumerate(test_x):
		qtem = qtem.reshape(1, -1)
		predicted = cluster_finder.predict(qtem)[0]	
		print('predicted=', predicted)
#		closest = find_closest(qtem, clustered_x[predicted], clustered_y[predicted], 'mutual_info_score')	
#		closest = find_closest(qtem, clustered_x[predicted], clustered_y[predicted], 'euclidean')			
		closest = find_closest(qtem, clustered_x[predicted], clustered_y[predicted], 'cosine')			
		print('closest=', closest)		
		generated =decode_sequence(closest[1])
		print('generated=', generated)				
		actual = decode_sequence(test_y[q])
		print('actual=', actual)						
		import nltk

		BLEUscore = nltk.translate.bleu_score.sentence_bleu([generated], actual)
		
		log = log_str + 'bleu-4=' + str(BLEUscore)		
		rouge = Rouge()
		scores = rouge.get_scores(generated, actual)
		log = log_str + str(scores)
		log_file.writelines(log+"\n")
		log_file.writelines("---------------------"+'\n')
		


log_file = open('hp-log-pca.txt', 'w')
log_file.writelines("---------------------"+'\n')

co = [50]

for eleme in co:

	clusters = eleme
	components = 800
	print(clusters, components)

	X_embedded = PCA(n_components=components).fit_transform(x)
	x = X_embedded

	test_X_embedded = PCA(n_components=components).fit_transform(test_x)
	test_x = test_X_embedded

	eval_batch(x, y, test_x, test_y, "mbk", components, clusters, "pca")
log_file.close()




print("--- %s seconds ---" % (time.time() - start_time))
sys.exit(0)




