import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

class1 = pd.read_csv("D:\\feature\\Class2_0_all_fet.csv")

class2 = pd.read_csv("D:\\feature\\Class2_10_all_fet.csv")

class3 = pd.read_csv("D:\\feature\\Class2_80_all_fet.csv")

result_combined = np.vstack((class1,class2,class3))

print(result_combined.shape)


a = class1.shape[0]
b = class2.shape[0]
c = class3.shape[0]
label = np.array([0] * a + [1] * b + [2] * c)  
print(label.shape)

tsne = TSNE(n_components=3, perplexity=20, init='pca', n_iter=300, random_state=42)
X_tsne = tsne.fit_transform(result_combined)

# # Perform K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_tsne)

# Create a DataFrame for the t-SNE results and clusters
# tsne_df = pd.DataFrame(X_tsne, columns=['TSNE1', 'TSNE2'])
# tsne_df['Cluster'] = label
plt.scatter(X_tsne[:, 0],X_tsne[:, 2], 15, clusters)
plt.show()
