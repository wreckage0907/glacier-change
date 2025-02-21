import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
import seaborn as sns

# Keywords and their frequencies in glacier change detection literature
keyword_freq = {
    "deep learning": 25, "glacier mapping": 22, "change detection": 20,
    "satellite imagery": 18, "debris-covered glaciers": 15, "CNN": 14,
    "multi-temporal": 13, "SAR data": 12, "Sentinel-2": 12,
    "feature extraction": 11, "glacier boundaries": 10, "U-Net": 10,
    "data fusion": 9, "transformer models": 9, "attention mechanism": 8,
    "transfer learning": 8, "glacier velocity": 7, "real-time monitoring": 7,
    "Landsat": 7, "ground truth": 6, "DEM": 6,
    "cloud computing": 5, "time series analysis": 5, "glacier retreat": 5,
    "image segmentation": 8, "GLIMS": 4, "mass balance": 6,
    "data augmentation": 7, "accuracy assessment": 8,
    "remote sensing": 16, "machine learning": 12
}


# Density Visualization Fixes
plt.figure(figsize=(14, 10))

positions = np.random.uniform(-2, 2, (len(keyword_freq), 2))
keyword_positions = {k: pos for k, pos in zip(keyword_freq.keys(), positions)}

x, y = positions[:, 0], positions[:, 1]
kernel = gaussian_kde(np.vstack([x, y]), bw_method=0.2)
xi, yi = np.mgrid[x.min():x.max():100j, y.min():y.max():100j]
zi = kernel(np.vstack([xi.flatten(), yi.flatten()])).reshape(xi.shape)

plt.imshow(zi.T, extent=[x.min(), x.max(), y.min(), y.max()],
           origin='lower', cmap='YlGnBu', alpha=0.6, aspect='auto')

for keyword, pos in keyword_positions.items():
    plt.text(pos[0], pos[1], keyword, 
             fontsize=8,
             ha='center', va='center',
             color='black', fontweight='bold', bbox=dict(facecolor='white', alpha=0.6))

plt.axis('off')
plt.show()

# Co-occurrence Network Fixes
G = nx.Graph()

for keyword, freq in keyword_freq.items():
    G.add_node(keyword, weight=freq)

keywords = list(keyword_freq.keys())
for i, kw1 in enumerate(keywords):
    for kw2 in keywords[i+1:]:
        common_words = set(kw1.split()) & set(kw2.split())
        if common_words or any(w in kw1 or w in kw2 for w in ["ocean", "marine", "coral", "pH"]):
            weight = (keyword_freq[kw1] + keyword_freq[kw2]) / 2
            G.add_edge(kw1, kw2, weight=weight)

pos = nx.spring_layout(G, k=0.5, seed=42, iterations=100)
plt.figure(figsize=(12, 8))
edge_weights = [G[u][v]['weight'] / 10 for u, v in G.edges()]

nx.draw_networkx_edges(G, pos, alpha=0.7, width=edge_weights, edge_color='gray')
nx.draw_networkx_nodes(G, pos, node_size=[keyword_freq[n] * 50 for n in G.nodes()], 
                      node_color='lightblue', alpha=0.9, edgecolors='black')
nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

plt.axis('off')
plt.title("Keyword Co-occurrence Network")
plt.show()
