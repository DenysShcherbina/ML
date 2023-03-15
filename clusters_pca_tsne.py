import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from scipy.stats import mode
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix


digits = load_digits()

# Create models and predict clusters
tsne = TSNE(n_components=2, init='pca', random_state=0)
digits_proj = tsne.fit_transform(digits.data)
kmeans = KMeans(n_clusters=10, random_state=0)
clusters = kmeans.fit_predict(digits_proj)
labels = np.zeros_like(clusters)
for i in range(10):
    mask = (clusters == i)
    labels[mask] = mode(digits.target[mask], keepdims=True)[0]

# Check metrics
matrix = confusion_matrix(digits.target, labels)
sns.heatmap(matrix.T, square=True, annot=True, fmt='d', cbar=False, xticklabels=digits.target_names,
            yticklabels=digits.target_names)
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.show()

