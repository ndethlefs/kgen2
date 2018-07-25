import sys


train_file = open("./wikipedia-biography-dataset-master/wikipedia-biography-dataset/train/train.sent", 'r')
test_file = open("./wikipedia-biography-dataset-master/wikipedia-biography-dataset/test/test.sent", 'r')
vocab_file = open("wiki_vocab.txt", "w")

for line in train_file:
	vocab_file.writelines(line)
for line in test_file:	
	vocab_file.writelines(line)	


train_file.close()
test_file.close()
vocab_file.close()



