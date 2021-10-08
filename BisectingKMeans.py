# %%
from collections import Counter
from scipy.sparse import csr_matrix
from collections import defaultdict


def csr_info(mat, name="", non_empy=False):
    r""" Print out info about this CSR matrix. If non_empy, 
    report number of non-empty rows and cols as well
    """
    if non_empy:
        print("%s [nrows %d (%d non-empty), ncols %d (%d non-empty), nnz %d]" % (
                name, mat.shape[0], 
                sum(1 if mat.indptr[i+1] > mat.indptr[i] else 0 
                for i in range(mat.shape[0])), 
                mat.shape[1], len(np.unique(mat.indices)), 
                len(mat.data)))
    else:
        print( "%s [nrows %d, ncols %d, nnz %d]" % (name, 
                mat.shape[0], mat.shape[1], len(mat.data)) )
        
# # scale matrix and normalize its rows
# def csr_idf(mat, copy=False, **kargs):
#     r""" Scale a CSR matrix by idf. 
#     Returns scaling factors as dict. If copy is True, 
#     returns scaled matrix and scaling factors.
#     """
#     if copy is True:
#         mat = mat.copy()
#     nrows = mat.shape[0]
#     nnz = mat.nnz
#     ind, val, ptr = mat.indices, mat.data, mat.indptr
#     # document frequency
#     df = defaultdict(int)
#     for i in ind:
#         df[i] += 1
#     # inverse document frequency
#     for k,v in df.items():
#         df[k] = np.log(nrows / float(v))  ## df turns to idf - reusing memory
#     # scale by idf
#     for i in range(0, nnz):
#         val[i] *= df[ind[i]]
        
#     return df if copy is False else mat

# def csr_l2normalize(mat, copy=False, **kargs):
#     r""" Normalize the rows of a CSR matrix by their L-2 norm. 
#     If copy is True, returns a copy of the normalized matrix.
#     """
#     if copy is True:
#         mat = mat.copy()
#     nrows = mat.shape[0]
#     nnz = mat.nnz
#     ind, val, ptr = mat.indices, mat.data, mat.indptr
#     # normalize
#     for i in range(nrows):
#         rsum = 0.0    
#         for j in range(ptr[i], ptr[i+1]):
#             rsum += val[j]**2
#         if rsum == 0.0:
#             continue  # do not normalize empty rows
#         rsum = 1.0/np.sqrt(rsum)
#         for j in range(ptr[i], ptr[i+1]):
#             val[j] *= rsum
            
#     if copy is True:
#         return mat

def build_matrix_1(docs, labels):
    r""" Build sparse matrix from a list of documents, 
    each of which is a list of word/terms in the document.  
    """
        
    distinctWordsAndIndex = {}
    indexIter = 0
    nnz = 0
    
    for idx, doc in enumerate(docs):
        frequency = doc.split()
        while frequency:
            term, freq, *frequency = frequency
#             print(term)
            if term not in labels:
                continue
            
            nnz += 1
            if term not in distinctWordsAndIndex:
                distinctWordsAndIndex[term] = indexIter
                indexIter += 1    
    nrows = len(docs)
    ncols = len(distinctWordsAndIndex)
    
    # set up memory
    ind = np.zeros(nnz, dtype=int)
    val = np.zeros(nnz, dtype=int)
    ptr = np.zeros(nrows+1, dtype=int)

    i = 0
    j = 0
    for idx, doc in enumerate(docs):
        ptr[j] = i
        j += 1
        frequency = doc.split()
        while frequency:
            term, freq, *frequency = frequency
            if term not in labels:
                continue
            ind[i] = distinctWordsAndIndex[term]
            val[i] = int(freq)
            i += 1
    ptr[j] = i

    mat = csr_matrix((val, ind, ptr), shape=(nrows, ncols), dtype=np.double)
    mat.sort_indices()
    
    return mat

from sklearn.cluster import KMeans
from numpy import *
%matplotlib inline
import matplotlib.pyplot as plt


def BisectingKMeans(mat4, k_start, k_end, step):
    for k in range(k_start, k_end+1, step):
#         print('============================================')
        X = mat4
        num_clusters = k
        k_list = [] 
        sse_list = [] 
        total_SSE = 0
        current_clusters = 1

        clusterMap = {}
        for idx,row in enumerate(X):
            clusterMap[idx] = idx
        finalClusterLabels = {}

        while current_clusters != num_clusters:
        #     print('final labels', finalClusterLabels)
        #     print('clusterMap',clusterMap)
            kmeans = KMeans(n_clusters=2).fit(X)
    #         print(kmeans.inertia_ )
            cluster_centers = kmeans.cluster_centers_
        #     print(X.shape)
            sse = [0]*2
            for point, label in zip(X, kmeans.labels_):
                sse[label] += np.square(point-cluster_centers[label]).sum()
            chosen_cluster = np.argmax(sse, axis=0)
            total_SSE += sse[np.argmin(sse, axis=0)]
    #         print('SSE', sse)
    #         print('Total SSE', total_SSE)
    #         print('chosen_cluster', chosen_cluster)
    #         print('kmeans labels', kmeans.labels_)
        #     print('cluster_centers', cluster_centers.shape)
            chosen_cluster_data = X[kmeans.labels_ == chosen_cluster]
    # 
            newClusterMap = {}
            clusterIter = 0
            for idx, x in enumerate(kmeans.labels_):
                if(x != chosen_cluster):
                    finalClusterLabels[clusterMap[idx]] = current_clusters
                elif current_clusters + 1 == num_clusters:
                    finalClusterLabels[clusterMap[idx]] = current_clusters + 1
                else:
                    newClusterMap[clusterIter] = clusterMap[idx]
                    clusterIter += 1 
            clusterMap = newClusterMap
            current_clusters += 1

    #         print('chosen_cluster_data', chosen_cluster_data.shape)
            assigned_cluster_data = X[kmeans.labels_ != chosen_cluster]
    #         print('assigned_cluster_data', assigned_cluster_data.shape)
            X = chosen_cluster_data
    #         print('finalClusterLabels - len ', len(finalClusterLabels))

        k_list.append(k)
        sse_list.append(kmeans.inertia_ )
        print_internal_metrics(mat4, finalClusterLabels)

    return finalClusterLabels
#         print(k_list, sse_list)


# %%
#pre-processing

import nltk
from nltk.corpus import stopwords

#Filter labels with length <= 3 and is present in stop words
stop_words = stopwords.words('english')
with open("train.clabel", "r", encoding="utf8") as fh:
    labels = {}
    for idx, word in enumerate(fh.readlines()):
        if len(word.rstrip()) < 4 or word.rstrip() in stop_words:
            continue
        labels[str(idx+1)] = word.rstrip()
# print(labels)


# %%
#Internal Metrics

from sklearn import metrics

def print_internal_metrics(mat, labels_dict):
    labels = [labels_dict[key] for key in sorted(labels_dict.keys())]
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    print(n_clusters_, metrics.silhouette_score(mat, labels), metrics.calinski_harabasz_score(mat, labels))
#     print('Estimated number of clusters: %d' % n_clusters_)
#     print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(mat, labels))
#     print("Calinski Harabasz Score: %0.3f" % metrics.calinski_harabasz_score(mat, labels))


# %%
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer 
from datetime import datetime


with open("train.dat", "r", encoding="utf8") as fh:
    rows = fh.readlines()

mat1 = build_matrix_1(rows, labels)

tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True) 
tf_idf_vector=tfidf_transformer.fit(mat1).transform(mat1)
print('Shape before SVD')
csr_info(tf_idf_vector)

print("Start Time =", datetime.now().strftime("%H:%M:%S"))
components = 5000
while components < 6001:
    print('============================================')
    print('SVD number of concepts = ', components)
    tsvd = TruncatedSVD(n_components=components)
    mat4 = tsvd.fit(tf_idf_vector).transform(tf_idf_vector)
    print('Variance ratio sum', tsvd.explained_variance_ratio_.sum())
#     print(''tsvd.singular_values_.sum())
#     print('Shape after SVD')
    csr_info(mat4)
#     finalClusterLabels = BisectingKMeans(mat4, 11, 11, 1)
    components += 500
print("End Time =", datetime.now().strftime("%H:%M:%S"))


# %%
csr_info(mat4)
finalClusterLabels = BisectingKMeans(mat4, 14, 14, 1)

# %%
# plt.plot(k_list, sse_list)
# plt.ylabel('SSE')
# plt.xlabel('k')
# plt.show()

# %%

labels = [finalClusterLabels[key] for key in sorted(finalClusterLabels.keys())]
with open("output.dat", "w", encoding="utf8") as file:
     for item in labels:
        file.write("%s\n" % str(item))
len(rows)

# %%
count_labels = {}
for label in labels:
    
    if label not in count_labels:
        count_labels[label] = 1
    else:
        count_labels[label] = int(count_labels[label]) + 1
#     print(count_labels[label])
print(count_labels)