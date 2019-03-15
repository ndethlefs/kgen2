import os,sys
from utils.config import *
from models.Mem2Seq_NMT import Mem2Seq
import numpy as np
import logging 
from tqdm import tqdm
from utils.utils_wiki import prepare_data_seq

train_dir = "./data/100_clusters1/all_train/"
test_dir = "./data/100_clusters1/all_test/"
out_dir = "./data/100_clusters1/out/"
#train_dir = "./data/100_test/all_train/"
#test_dir = "./data/100_test/all_test/"
#out_dir = "./data/100_test/out/"
#train_dir = "./data/webnlg_50_clusters/all_train/"
#test_dir = "./data/webnlg_50_clusters/all_test/"
#out_dir = "./data/webnlg_50_clusters/out/"
#train_dir = "./data/webnlg3/all_train/"
#test_dir = "./data/webnlg3/all_test/"
#out_dir = "./data/webnlg3/out/"
files = os.listdir(train_dir)

for f in files:
	
	train,test,lang, max_len, max_r = prepare_data_seq(train_dir+f,test_dir+f,batch_size = 32)

	model = Mem2Seq(hidden_size= 256, max_len= max_len, 
    	            max_r= max_r, lang=lang, 
        	        path="",lr=0.001, n_layers=3, dropout=0.2)

	epochs=5000
	out_file = open(out_dir+f, "w")

	avg_best = 0
	for epoch in range(epochs):
		logging.info("Epoch:{}".format(epoch))  
	    # Run the train function
		pbar = tqdm(enumerate(train),total=len(train))
		for i, data in pbar:
			model.train_batch(input_batches=data[0], 
        	                  input_lengths=data[1], 
            	              target_batches=data[2], 
                	          target_lengths=data[3], 
                    	      target_index=data[4], 
                        	  batch_size=len(data[1]),
	                          clip= 10.0,
    	                      teacher_forcing_ratio=0.5,
        	                  reset=False)

			pbar.set_description(model.print_loss())

#	if((epoch+1) % 1 == 0):
		if((epochs+1) % 1 == 0):	
			bleu = model.evaluate(out_file,train,avg_best)
			model.scheduler.step(bleu)
			if(bleu >= avg_best):
				avg_best = bleu
				cnt=0
			else:
				cnt+=1
		
			if(cnt == 5): break

	out_file.close()