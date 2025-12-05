import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import svm
import seaborn as sns

# Loading the dataset 
df = pd.read_csv('falldetection_dataset.csv',header=None)
df = df.sample(frac = 1)

# Looking the heads 
print(df.head())

# Droping the first column 
df = df.drop(columns=[0])
print(df.shape)

# Arranging Dataset into x and y 
dataset_y = df[1]
dataset_x = df.drop(columns=[1])

# One hot encoding 
dataset_y_changed = dataset_y.replace({"F":1,"NF":0})

# Centering the data each column 

features_dataset_mean = dataset_x.mean(axis=0)
features_dataset_std = dataset_x.std(axis=0)
X_centered =( dataset_x - features_dataset_mean) / features_dataset_std

# Applying PCA 
n_features = X_centered.shape[1]
pca = PCA(n_components=n_features)
pca.fit(X_centered)
23



eigenvalues_pca = pca.explained_variance_

plt.figure()
plt.plot(range(1,n_features+ 1), eigenvalues_pca,marker='o')
plt.xlabel('Principal Component')
plt.ylabel('Eigenvalue')
plt.title('Eigenvalues in Descending Order')

