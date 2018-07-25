import sys
import numpy as np
import bleu
import nltk


infile = open('hp-log-pca.txt', 'r')

bleu = []
rouge = []

for line in infile:
	print(line)
	if "bleu-4" in line:
		b = str(line.split("bleu-4=")[1])
		b = float(b[0:10])
		print(b)
		bleu.append(b)

	if "rouge-l" in line:
		b = str(line.split("rouge-l': {'")[1])
		b = b.split("'p': ")[1]
		print(b)
		if "}}]" in b:
			b = float(b.split("}}]")[0])
		elif "," in b:
			b = float(b.split(",")[0])				
	
		
		rouge.append(b)		
		
print(len(bleu))
bleu = np.asarray(bleu)
bleu_mean = np.mean(bleu)
print(bleu)
print('bleu_mean=', bleu_mean)

print(len(rouge))
rouge = np.asarray(rouge)
rouge_mean = np.mean(rouge)
print('rouge_mean=', rouge_mean)

infile.close()

#sys.exit(0)

infile = open("wiki_pca.txt", 'r')
bleu =[]

generated = []
actual = []

for line in infile:
	if "generated=" in line:
		print(line)
		g = str(line.split("generated= ")[1]).replace("\n", "")
		generated = g.split()
		print(generated)
	if "actual=" in line:
		a = str(line.split("actual= ")[1]).replace("\n", "")
		actual = a.split()
		print(actual)
		if len(actual)>3:
			BLEUscore = nltk.translate.bleu_score.sentence_bleu([generated], actual)
			print(BLEUscore)
			bleu.append(BLEUscore)

bleu = np.asarray(bleu)
bleu_mean = np.mean(bleu)		
print('bleu_mean=', bleu_mean)

infile.close()
