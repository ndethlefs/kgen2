import sys
import time
import copy
import json

class TripletsReader:

	def __init__(self, input_file):
		self.input_file = input_file
		self.graph = {}
		self.graph_delex = {}
		self.training = []
		self.targets = {}


	def storeTriplets(self, filename):
		print("Reading",filename,"...")
		lines = [line.rstrip('\n') for line in open(filename)]

		counter=0
		for line in lines:
			triplet = line.split(',')
			key = triplet[0]
			_key = triplet[1]
			_val = triplet[2]
	
			self.storeTriplet(key, _key, _val)
			#storeTriplet(_val, _key, key)
			counter=counter+1
		
			if counter % 10000 == 0:
				print("Processed", len(self.graph), "triplets...")
	
	def storeTriplet(self, key, _key, _val):
			subgraph = {}
			list = []

			if key in self.graph:
				subgraph = self.graph[key]

				if _key in subgraph:
					list = subgraph[_key]
			
			list.append(_val)
			subgraph[_key] = list
			self.graph[key] = subgraph	

	def printDictionary(self):
		for key in  graph:
			subgraph = graph[key]
			for _key in subgraph:
				_val = subgraph[_key]
				print(key, _key, _val)


	def query(self, target):
		print("=== Querying ",target)
		for key in self.graph:
			if key == target:
				#print "key=",key
				subgraph = self.graph[key]
				print("_key=",subgraph)
				
				
	def delexicalise_graph(self):
		graph_delex = copy.deepcopy(reader.graph)
		for d in graph_delex:
			for e in graph_delex[d]:
				graph_delex[d][e] = [e]
			#for key in reader.graph:
				#print(key, reader.graph[key])
				#print(key, graph_delex[key])
		self.graph_delex = graph_delex								
		return graph_delex	
			
			
	def get_text(self, file):
	
		# get titles for all articles
		for line in open(file+"test.title",'r'):
			title = line.replace("\n", "")
			self.training.append([title])
#		print(self.training)	
		
		# get number of sentences for all articles
		line_number = 0
		for line in open(file+"test.nb","r"):
			nb = int(line.replace("\n", ""))
#			print(line_number)
#			print(self.training)			
			self.training[line_number].append(nb)
			line_number = line_number + 1
#		print(self.training)
		
		# add first sentence for each
		
		all_sents = []
		for line in open(file+"test.sent", 'r'):
			all_sents.append(line.replace("\n", ""))
		
		current_line = 0
		for t in self.training:
			para = all_sents[current_line]
			current_line = current_line + t[1]
			t.append(para)
		
		for t in self.training:
			if t[0] in self.graph:
				semantics = self.graph[t[0]]	
				for s in semantics:
					att = s
					val = semantics[s]
					if val[0] in t[2]:
						t[2] = t[2].replace(" "+val[0]+" ", " "+att+" ")

			self.targets[t[0]] = t[2]		

		print('targets', self.targets)
		

	def load_targets(self, file):
		
		json_data=open(file).read()

		data = json.loads(json_data)
		self.targets = data
#		print(self.targets)
		

	def load_d_graph(self, file):
		
		json_data=open(file).read()

		data = json.loads(json_data)
		self.graph_delex = data
#		print(self.targets)		
		

	def save_targets(self, file):
	
		with open(file, 'w') as fp:
		    json.dump(self.targets, fp)	
			
	def save_d_graph(self, file):
	
		with open(file, 'w') as fp:
		    json.dump(self.graph_delex, fp)				
			
			
'''def storeMoreData(filename):
	lines = [line.rstrip('\n') for line in open(filename)]

	for line in lines:
		triplet = line.split(',')
		key = triplet[2]
		_key = triplet[1]
		_val = triplet[0]
		list = []
	
		if not key in graph:
			subgraph = {}
		
		else:
			subgraph = graph[key]
			if _key in subgraph:
				list = subgraph[_key]
			
		list.append(_val)
		subgraph[_key] = list
		graph[key] = subgraph	
	
		print triplet
	print graph'''


start_time = time.time()

#reader = TripletsReader('wiki-triplets-test.txt')
#reader.storeTriplets('wiki-triplets-test.txt')
#graph_delex = reader.delexicalise_graph()
#reader.load_targets('result.json')
#reader.get_text("./wikipedia-biography-dataset-master/wikipedia-biography-dataset/test/")
#print('target',reader.targets["lenny randle"])
#reader.save_targets('targets-test.json')
#print(reader.targets)
#reader.save_d_graph('d_graph-test.json')
#reader.load_d_graph('d_graph-test.json')
#print('delex',reader.graph_delex["lenny randle"])
#print(reader.graph_delex)
#printDictionary()
#reader.query("aaron hohlbein")
#query("USA")
#query("walter extra")
#query("Will Smith")
#query("Simon Cowell")


#graph_delex = reader.delexicalise_graph()
#print(graph_delex)


print("--- %s seconds ---" % (time.time() - start_time))

#dir = "./wikipedia-biography-dataset-master/wikipedia-biography-dataset/train/"
#reader.get_text(dir)


#reader.get_indices(dir+"train.title")
#reader.get_sents(dir+"train.nb")
#reader.get_text(dir)
#reader.delex_text()
#reader.get_targets()
	
#print(reader.target['walter extra'])	
#print(reader.targets["walter extra"])
#print("---------")
#print(reader.targets["aaron hohlbein"])


print("---------")









