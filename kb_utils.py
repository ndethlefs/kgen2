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
from sklearn.metrics import mutual_info_score
from scipy.stats import entropy
from scipy.spatial import distance

start_time = time.time()

def sort_data(graph):
	
	sorted_data = {}
	for key in graph:
		sorted_data[key] = sorted(graph[key])

	return sorted_data

	for s in sorted_data:
		x.append(" ".join(sorted_data[s]))
		y.append(s) 	
		print(x,y)
		if len(sorted_data[s]) > max_len:
			max_len = len(sorted_data[s])


def get_x_y(sorted_dict):
	
	x = []
	y = []
	
	for s in sorted_dict:
		x.append(" ".join(sorted_dict[s]))
		y.append(s) 	
		print(x,y)
	x = np.asarray(x)
	y = np.asarray(y)		
	
	return x, y	


def decode_sequence(seq):
	decoded = ""
	for s in seq:
		if not s==0:
			decoded = decoded + index_word[int(s)] + " "
	return decoded.strip()	


def get_max_len(x, y):

	max_len = 0
	for item in x:
		if len(item) > max_len:
			max_len = len(item)
	for item in y:
		if len(item) > max_len:
			max_len = len(item)
	return max_len		


def KL(a, b):
    a = np.asarray(a, dtype=np.float)
    b = np.asarray(b, dtype=np.float)

    return np.sum(np.where(a != 0, a * np.log(a / b), 0))
    
    
def find_closest(p, data, labels, metric='euclidean'):

	# metric can be mutual_info_score, KL, entropy or euclidean
	
	closest = []
	label = []
	curr_distance = 50000
	
	if metric=='mutual_info_score':
		for i, item in enumerate(data):
			if mutual_info_score(p, item) < curr_distance:
				closest = item
				label = labels[i]
				curr_distance = mutual_info_score(p, item)
	elif metric=='kl':
		for i, item in enumerate(data):
			if KL(p, item) < curr_distance:
				closest = item
				label = labels[i]
				curr_distance = KL(p, item)
				print(KL(p, item))				
				print(curr_distance)				
	elif metric=='entropy':
		for i, item in enumerate(data):
			if entropy(p, item) < curr_distance:
				closest = item
				label = labels[i]
				curr_distance = entropy(p, item)
	elif metric=='cosine':
		for i, item in enumerate(data):
			if distance.cosine(p, item) < curr_distance:
				closest = item
				label = labels[i]
				curr_distance = distance.cosine(p, item)					
	else:
		for i, item in enumerate(data):
			if distance.euclidean(p, item) < curr_distance:
				closest = item
				label = labels[i]
				curr_distance = distance.euclidean(p, item)

	return closest, label, curr_distance    
	
	
	
	
	