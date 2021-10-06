# %%
from collections import Counter
from scipy.sparse import csr_matrix
from collections import defaultdict

def build_matrix_1(docs):
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
            ind[i] = distinctWordsAndIndex[term]
            val[i] = int(freq)
            i += 1
    ptr[j] = i

    mat = csr_matrix((val, ind, ptr), shape=(nrows, ncols), dtype=np.double)
    mat.sort_indices()
    
    return mat


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
        
# scale matrix and normalize its rows
def csr_idf(mat, copy=False, **kargs):
    r""" Scale a CSR matrix by idf. 
    Returns scaling factors as dict. If copy is True, 
    returns scaled matrix and scaling factors.
    """
    if copy is True:
        mat = mat.copy()
    nrows = mat.shape[0]
    nnz = mat.nnz
    ind, val, ptr = mat.indices, mat.data, mat.indptr
    # document frequency
    df = defaultdict(int)
    for i in ind:
        df[i] += 1
    # inverse document frequency
    for k,v in df.items():
        df[k] = np.log(nrows / float(v))  ## df turns to idf - reusing memory
    # scale by idf
    for i in range(0, nnz):
        val[i] *= df[ind[i]]
        
    return df if copy is False else mat

def csr_l2normalize(mat, copy=False, **kargs):
    r""" Normalize the rows of a CSR matrix by their L-2 norm. 
    If copy is True, returns a copy of the normalized matrix.
    """
    if copy is True:
        mat = mat.copy()
    nrows = mat.shape[0]
    nnz = mat.nnz
    ind, val, ptr = mat.indices, mat.data, mat.indptr
    # normalize
    for i in range(nrows):
        rsum = 0.0    
        for j in range(ptr[i], ptr[i+1]):
            rsum += val[j]**2
        if rsum == 0.0:
            continue  # do not normalize empty rows
        rsum = 1.0/np.sqrt(rsum)
        for j in range(ptr[i], ptr[i+1]):
            val[j] *= rsum
            
    if copy is True:
        return mat

# %%
import numpy as np

with open("train.dat", "r", encoding="utf8") as fh:
    rows = fh.readlines()
len(rows)

# rows = ['1 1 2 1 3 1', 
#         '1 2 2 1 4 1', 
#         '1 1 2 1 6 1', 
#         '1 1 2 1 5 1', 
#         '1 1 2 1 7 1',
#        '1 1 2 1 8 1', 
#         '1 1 2 1 9 1']

mat1 = build_matrix_1(rows)
# csr_info(mat1)
mat2 = csr_idf(mat1, copy=True)
mat3 = csr_l2normalize(mat2, copy=True)

print(mat1.shape)
# print("mat1:", mat1[15,:20].todense(), "\n")
# print("mat2:", mat2[15,:20].todense(), "\n")
# print("mat3:", mat3[15,:20].todense())

# %%
from sklearn.cluster import KMeans
from numpy import *

num_clusters = 7

X = mat3
clusterMap = {}
for idx,row in enumerate(X):
    clusterMap[idx] = idx

# print(localToMatMapping)

finalClusterLabels = {}

# labels = 
X = mat3
current_clusters = 1
while current_clusters != num_clusters:
    print('============================================')
#     print('final labels', finalClusterLabels)
#     print('clusterMap',clusterMap)
    kmeans = KMeans(n_clusters=2).fit(X)
    cluster_centers = kmeans.cluster_centers_
#     print(X.shape)
    sse = [0]*2
    for point, label in zip(X, kmeans.labels_):
        sse[label] += np.square(point-cluster_centers[label]).sum()
    chosen_cluster = np.argmax(sse, axis=0)
#     print(sse)
    print('chosen_cluster', chosen_cluster)
    print('kmeans labels', kmeans.labels_)
#     print('cluster_centers', cluster_centers.shape)
    chosen_cluster_data = X[kmeans.labels_ == chosen_cluster]
    
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
    
    print('chosen_cluster_data', chosen_cluster_data.shape)
    assigned_cluster_data = X[kmeans.labels_ != chosen_cluster]
    print('assigned_cluster_data', assigned_cluster_data.shape)
    X = chosen_cluster_data
    print('finalClusterLabels - len ', len(finalClusterLabels))

# print(finalClusterLabels)

# print(mat3)
# km = KMeans(n_clusters=num_clusters)
# km.fit(mat3)

# %%
#Compare with last output

from sklearn import metrics
from sklearn.metrics.cluster import normalized_mutual_info_score

with open("output.dat", "r", encoding="utf8") as fh:
    labels_true = fh.readlines()

# Number of clusters in labels, ignoring noise if present.
labels = [finalClusterLabels[key] for key in sorted(finalClusterLabels.keys(), reverse=True)]
# labels = km.labels_
# labels_true =  km.labels_.tolist()

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
print('Estimated number of clusters: %d' % n_clusters_)
print('NMI: %s' % normalized_mutual_info_score(labels, labels_true))
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
print("Adjusted Rand Index: %0.3f"
 % metrics.adjusted_rand_score(labels_true, labels))
print("Adjusted Mutual Information: %0.3f"
 % metrics.adjusted_mutual_info_score(labels_true, labels))
# print("Silhouette Coefficient: %0.3f"
#  % metrics.silhouette_score(X, labels))

# %%
with open("output.dat", "w", encoding="utf8") as file:
     for item in labels:
        file.write("%s\n" % str(item))
len(rows)