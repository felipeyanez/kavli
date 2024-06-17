"""
Max Planck Group: In Silico Brain Sciences
Center of Advanced European Studies and Research

Collection of functions used in the morphological classificaiton protocol.

Author: Felipe Yanez
Date: 09.03.2020    
"""

##########################################################################################
### LOAD LIBRARIES
##########################################################################################

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from scipy.io import loadmat
from scipy.stats import variation
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage, leaves_list
from sklearn.model_selection import KFold
# from fastcluster import linkage #as fc_linkage

# pip install git+https://github.com/kylessmith/dynamicTreeCut     
from dynamicTreeCut import cutreeHybrid

# # pip install git+https://github.com/AllenInstitute/drcme
# from helpers.drcme.ephys_morph_clustering import consensus_clusters

##########################################################################################
### LOAD DATA
##########################################################################################

# Load morphological features
def load_features(filename):
	# Read features
	morph_features = loadmat(filename)
	# Read names of morphological features
	featureIDs = morph_features['featureIDs']
	featureIDs = [featureIDs[0,i][0] for i in range(featureIDs.size)]
	# Analize separately the cells with axon only
	X = pd.DataFrame(morph_features['features'], columns=featureIDs)
	# Find cells with missing either axon or dendrites reconstructions
	invalid_cells = np.isnan(morph_features['features'].sum(1))
	return X, invalid_cells

# Load evaluations by experts
def load_evals(filename):
	# Read evaluations
	evals = loadmat(filename)
	evals = evals['evaluations']
	# Read ids because data is shuffled
	ids = [evals[i][0][0][0][1][0] for i in range(len(evals))]
	# For the moment return first classification
	type1 = [evals[i][0][0][0][2][0] for i in range(len(evals))]
	# Create data frame to sort ids
	df = pd.DataFrame({'Cell ID': ids, 'Type 1': type1}).set_index('Cell ID')
	return df.sort_values(by=['Cell ID'])


##########################################################################################
### PRE-PROCESSING FEATURES
##########################################################################################

# Drop features with low variance and highly correlated
def drop_features(X, a, b):
	v = unique_idx(a, b)
	return X.drop(X.columns[v], axis=1)

# Find unique indices of 2 vectors
def unique_idx(a, b):
	a.extend(i for i in b if i not in a)
	return a

# Find indices of features with low variance (CV < 0.25)
def low_cv(X, CV=0.25):
	cv_X = pd.DataFrame(abs(variation(X, axis=0))).T
	return [column for column in cv_X.columns if any(cv_X[column] < CV)]

# Find indices of highly correlated features (correlation > 0.95)
def high_corr(X, corr_value=0.95):
	corr_X = X.corr().abs()
	U = corr_X.where(np.triu(np.ones(corr_X.shape), k=1).astype(np.bool))
	U.columns = range(len(U.columns))
	return [column for column in U.columns if any(U[column] > corr_value)]

# Count number of cells in each cluster
def count_per_cluster(labels):
	tmp = pd.DataFrame({'Number of cells':range(len(labels)), 'Cluster':labels})
	return pd.DataFrame(tmp.groupby('Cluster')['Number of cells'].nunique()).T


##########################################################################################
### CLUSTERING
##########################################################################################

# Iterative clustering with consensus and merge of classes
def iterative_hc(X, n_rounds=0, n_folds=10, shuffle=True, 
	metric='euclidean', method='ward', minClusterSize=1, verbose=0):
	df = pd.DataFrame({'Cell ID': X.index}).set_index('Cell ID')
	for i in range(n_rounds):
		df = run_single(df, X, current_round=i, n_splits=n_folds, 
			shuffle=True, metric='euclidean', method='ward', 
			minClusterSize=1, verbose=0)
	labels, shared_norm = consensus_clusters(df.values)
	# TO-DO: merge classes
	return remap_labels(X, labels), shared_norm

# One clustering round in a k-fold manner
def run_single(df, X, current_round=0, n_splits=10, shuffle=True, 
	metric='euclidean', method='ward', minClusterSize=1, verbose=0):
	counter = 0
	kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=current_round)
	for eval_idx, _ in kf.split(X.index):
		key = 'iter{:d}_split{:d}'.format(current_round, counter)
		df.loc[eval_idx, key] = morph_hc(X.loc[eval_idx, :])
		counter += 1
	return df

# Hierarchical clustering with automatic tree cut
def morph_hc(X, metric='euclidean', method='ward', minClusterSize=1, verbose=0):
	distM = pdist(X, metric=metric)
	dendro = linkage(distM, method=method)
	model = cutreeHybrid(dendro, distM, minClusterSize=minClusterSize, verbose=verbose)   
	return remap_labels(X, model['labels'])

# Form a tree using data from multiple rounds and cut it to get a consensus
def consensus_clusters(results, method='ward', minClusterSize=1, verbose=0):
	n_cells = results.shape[0]
	shared = np.zeros((n_cells, n_cells))
	for i in range(shared.shape[0]):
		for j in range(i, shared.shape[0]):
			shared_count = np.sum(results[i, :] == results[j, :])
			shared[i, j] = shared_count
			shared[j, i] = shared_count
	shared_norm = shared / shared.max()
	dendro = linkage(shared_norm, method=method)
	model = cutreeHybrid(dendro, shared_norm, 
		minClusterSize=minClusterSize, verbose=verbose)  
	return model['labels'], shared_norm

def remap_labels(X, labels):
	# Compress features using PCA
	X_pca = pd.DataFrame(PCA(n_components=2).fit_transform(X))
	# Calculate class averages
	avg = X_pca.groupby(labels).mean()
	# Sort classes using dendrogram
	Z = linkage(pdist(avg), 'ward', optimal_ordering = True)
	new_order = list(leaves_list(Z)[::-1])
	# Remap classes using new order of labels
	df = pd.DataFrame({'labels': labels - labels.min()})
	return df['labels'].apply(lambda x: 1 + new_order.index(x)).values


##########################################################################################
### SAVING DATA
##########################################################################################

# Format labels such that all cells are included 
def all_cell_labels(labels, invalid_cells, value_invalid=-1):
	df = pd.DataFrame({'Cell ID': range(len(invalid_cells))}).set_index('Cell ID')
	df.loc[invalid_cells, 'Cluster'] = value_invalid
	df.loc[~invalid_cells, 'Cluster'] = labels
	return df.values

