import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D  

#_latest_checkpoint
name = 'yoochoose164_LSTM_ATT_latest_checkpoint_12_11_2024_01:31:02'
embeddings = torch.load('embeddings/'+name+'.pth.tar').cpu().numpy()
print(f"Embeddings shape: {embeddings.shape}")

# Initialize t-SNE
# tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
# embeddings_2d = tsne.fit_transform(embeddings)

# Reduce dimensionality to 50 components using PCA first
pca = PCA(n_components=50)
embeddings_reduced = pca.fit_transform(embeddings)
embeddings_2d = TSNE(n_components=3, random_state=42).fit_transform(embeddings_reduced)

# Set Seaborn style
# sns.set(style='whitegrid')

# Plotting the 3D scatter plot
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')

# Use the labels for color mapping (if available)
# scatter = ax.scatter(
#     embeddings_3d[:, 0], 
#     embeddings_3d[:, 1], 
#     embeddings_3d[:, 2], 
#     c=labels if 'labels' in locals() else 'blue', 
#     cmap='viridis', 
#     s=15, 
#     alpha=0.7
# )

# Adding labels and colorbar
# ax.set_xlabel('TSNE Component 1')
# ax.set_ylabel('TSNE Component 2')
# ax.set_zlabel('TSNE Component 3')
# plt.savefig(name)


# # Scatter plot with labels
plt.figure(figsize=(10, 8))
scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], s=10, alpha=0.7) #cmap='plasma', c=cluster_labels
plt.title('t-SNE Visualization of Embeddings with Labels')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.xlim(-50,50) 
plt.ylim(-50,50)
plt.colorbar(scatter)
plt.grid(True)
plt.savefig(name)
#plt.show()
print('hi')