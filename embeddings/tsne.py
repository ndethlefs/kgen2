#
#  tsne.py
#
# Implementation of t-SNE in Python. The implementation was tested on Python 2.7.10, and it requires a working
# installation of NumPy. The implementation comes with an example on the MNIST dataset. In order to plot the
# results of this example, a working installation of matplotlib is required.
#
# The example can be run by executing: `ipython tsne.py`
#
#
#  Created by Laurens van der Maaten on 20-12-08.
#  Copyright (c) 2008 Tilburg University. All rights reserved.

import numpy as Math
import pylab as Plot
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from pylab import *
import matplotlib.pyplot as plt

def Hbeta(D = Math.array([]), beta = 1.0):
	"""Compute the perplexity and the P-row for a specific value of the precision of a Gaussian distribution."""

	# Compute P-row and corresponding perplexity
	P = Math.exp(-D.copy() * beta);
	sumP = sum(P);
	H = Math.log(sumP) + beta * Math.sum(D * P) / sumP;
	P = P / sumP;
	return H, P;


def x2p(X = Math.array([]), tol = 1e-5, perplexity = 30.0):
	"""Performs a binary search to get P-values in such a way that each conditional Gaussian has the same perplexity."""

	# Initialize some variables
	print("Computing pairwise distances...")
	(n, d) = X.shape;
	sum_X = Math.sum(Math.square(X), 1);
	D = Math.add(Math.add(-2 * Math.dot(X, X.T), sum_X).T, sum_X);
	P = Math.zeros((n, n));
	beta = Math.ones((n, 1));
	logU = Math.log(perplexity);

	# Loop over all datapoints
	for i in range(n):

		# Print progress
		if i % 500 == 0:
			print("Computing P-values for point ", i, " of ", n, "...")

		# Compute the Gaussian kernel and entropy for the current precision
		betamin = -Math.inf;
		betamax =  Math.inf;
		Di = D[i, Math.concatenate((Math.r_[0:i], Math.r_[i+1:n]))];
		(H, thisP) = Hbeta(Di, beta[i]);

		# Evaluate whether the perplexity is within tolerance
		Hdiff = H - logU;
		tries = 0;
		while Math.abs(Hdiff) > tol and tries < 50:

			# If not, increase or decrease precision
			if Hdiff > 0:
				betamin = beta[i].copy();
				if betamax == Math.inf or betamax == -Math.inf:
					beta[i] = beta[i] * 2;
				else:
					beta[i] = (beta[i] + betamax) / 2;
			else:
				betamax = beta[i].copy();
				if betamin == Math.inf or betamin == -Math.inf:
					beta[i] = beta[i] / 2;
				else:
					beta[i] = (beta[i] + betamin) / 2;

			# Recompute the values
			(H, thisP) = Hbeta(Di, beta[i]);
			Hdiff = H - logU;
			tries = tries + 1;

		# Set the final row of P
		P[i, Math.concatenate((Math.r_[0:i], Math.r_[i+1:n]))] = thisP;

	# Return final P-matrix
#	print("Mean value of sigma: ", Math.mean(Math.sqrt(1 / beta));
	return P;


def pca(X = Math.array([]), no_dims = 50):
	"""Runs PCA on the NxD array X in order to reduce its dimensionality to no_dims dimensions."""

	print("Preprocessing the data using PCA...")
	(n, d) = X.shape;
	X = X - Math.tile(Math.mean(X, 0), (n, 1));
	(l, M) = Math.linalg.eig(Math.dot(X.T, X));
	Y = Math.dot(X, M[:,0:no_dims]);
	return Y;


def tsne(X = Math.array([]), no_dims = 2, initial_dims = 50, perplexity = 30.0):
	"""Runs t-SNE on the dataset in the NxD array X to reduce its dimensionality to no_dims dimensions.
	The syntaxis of the function is Y = tsne.tsne(X, no_dims, perplexity), where X is an NxD NumPy array."""

	# Check inputs
	if isinstance(no_dims, float):
#		print("Error: array X should have type float.";)
		return -1;
	if round(no_dims) != no_dims:
#		print("Error: number of dimensions should be an integer.";)
		return -1;

	# Initialize variables
	X = pca(X, initial_dims).real;
	(n, d) = X.shape;
	max_iter = 1000;
	initial_momentum = 0.5;
	final_momentum = 0.8;
	eta = 500;
	min_gain = 0.01;
	Y = Math.random.randn(n, no_dims);
	dY = Math.zeros((n, no_dims));
	iY = Math.zeros((n, no_dims));
	gains = Math.ones((n, no_dims));

	# Compute P-values
	P = x2p(X, 1e-5, perplexity);
	P = P + Math.transpose(P);
	P = P / Math.sum(P);
	P = P * 4;									# early exaggeration
	P = Math.maximum(P, 1e-12);

	# Run iterations
	for iter in range(max_iter):

		# Compute pairwise affinities
		sum_Y = Math.sum(Math.square(Y), 1);
		num = 1 / (1 + Math.add(Math.add(-2 * Math.dot(Y, Y.T), sum_Y).T, sum_Y));
		num[range(n), range(n)] = 0;
		Q = num / Math.sum(num);
		Q = Math.maximum(Q, 1e-12);

		# Compute gradient
		PQ = P - Q;
		for i in range(n):
			dY[i,:] = Math.sum(Math.tile(PQ[:,i] * num[:,i], (no_dims, 1)).T * (Y[i,:] - Y), 0);

		# Perform the update
		if iter < 20:
			momentum = initial_momentum
		else:
			momentum = final_momentum
		gains = (gains + 0.2) * ((dY > 0) != (iY > 0)) + (gains * 0.8) * ((dY > 0) == (iY > 0));
		gains[gains < min_gain] = min_gain;
		iY = momentum * iY - eta * (gains * dY);
		Y = Y + iY;
		Y = Y - Math.tile(Math.mean(Y, 0), (n, 1));

		# Compute current value of cost function
		if (iter + 1) % 10 == 0:
			C = Math.sum(P * Math.log(P / Q));
			print("Iteration ", (iter + 1), ": error is ", C)

		# Stop lying about P-values
		if iter == 100:
			P = P / 4;

	# Return solution
	return Y;

def randrange(n, vmin, vmax):
    return (vmax - vmin)*np.random.rand(n) + vmin
    
if __name__ == "__main__":
	print("Run Y = tsne.tsne(X, no_dims, perplexity) to perform t-SNE on your dataset.")
	print("Running example on 2,500 MNIST digits...")
	X = Math.loadtxt("t8-vectors1.txt");		
#	X = Math.loadtxt("mnist.txt");			
#	labels = Math.loadtxt("labels.txt");
	labels = Math.loadtxt("t8-labels1.txt");	
	Y = tsne(X, 3, 50, 20.0);
	print(len(labels))
	
	fig = Plot.figure()
	ax = fig.add_subplot(111, projection='3d', axisbg='dimgray')
	ax.xaxis.set_major_formatter(plt.NullFormatter())
	ax.yaxis.set_major_formatter(plt.NullFormatter())	
	ax.zaxis.set_major_formatter(plt.NullFormatter())	
#	ax.w_xaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
#	ax.w_yaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
#	ax.w_zaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))		
#	ax.grid(color='r')

	
	file1 = open('plotdata.txt', 'w')	
	
	for i in range(len(Y)):
		print(labels[i], Y[i])
		file1.writelines(str(labels[i]) + '===' + str(Y[i]) + '\n')
	
	file1.close()
		
#	ax.scatter(Y[:,0], Y[:,1], Y[:,2], s=100, alpha=0.9, c=labels, marker='o', cmap=cm.prism, edgecolor='white')
	ax.scatter(Y[:,0], Y[:,1], Y[:,2], s=120, alpha=0.8, c=labels, marker='o', cmap=cm.gist_rainbow, edgecolor='grey', linewidth=0.5)	
#	ax.scatter(Y[:,0], Y[:,1], Y[:,2], s=50, alpha=0.9, c=labels, marker='o', cmap=cm.prism, lw = 0)	
	Plot.legend(loc='upper left', numpoints=5, ncol=5, fontsize=8, bbox_to_anchor=(0, 0))
	


	Plot.grid(color='r')	    
	Plot.savefig('foo.png', bbox_inches='tight')
#	Plot.savefig('destination_path.eps', format='eps', dpi=1000, facecolor='black', edgecolor='black', axisbg='black')
#	Plot.show();
	




