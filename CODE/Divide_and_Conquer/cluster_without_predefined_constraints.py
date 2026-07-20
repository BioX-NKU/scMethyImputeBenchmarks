import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import euclidean_distances
from scipy.cluster.hierarchy import linkage, leaves_list
import io


df = pd.read_csv("top_12_2i_cells_basic_stats.csv")

# Feature Engineering and Weighting
df['coverage_log'] = np.log10(df['total_observed_cpg'])

X_raw_rate = df[['cell_methylation_rate']].values
X_log_cov = df[['coverage_log']].values

# Concatenate features
X_combined = np.hstack([X_raw_rate, X_log_cov])
weights = np.array([50, 1]) 
X_weighted = X_combined * weights

# 3. Hierarchical Clustering
cluster_model = AgglomerativeClustering(
    distance_threshold=5, 
    n_clusters=None, 
    linkage='ward'
)
labels = cluster_model.fit_predict(X_weighted)
df['cluster_initial'] = labels

# 4. Post-processing: Merge small clusters
def merge_small_clusters(X_data, labels, min_size):
    unique_labels = np.unique(labels)
    if len(unique_labels) <= 1: return labels
    
    df_temp = pd.DataFrame(X_data)
    df_temp['label'] = labels
    counts = df_temp['label'].value_counts()
    
    if counts.max() < min_size: return labels 

    new_labels = labels.copy()
    valid_labels = counts[counts >= min_size].index.values
    valid_centroids = df_temp[df_temp['label'].isin(valid_labels)].groupby('label').mean().values
    valid_lbl_map = df_temp[df_temp['label'].isin(valid_labels)].groupby('label').mean().index.values
    
    for label in unique_labels:
        if counts[label] < min_size:
            indices = np.where(labels == label)[0]
            for idx in indices:
                sample = X_data[idx].reshape(1, -1)
                dists = euclidean_distances(sample, valid_centroids)
                new_labels[idx] = valid_lbl_map[np.argmin(dists)]
    return new_labels

min_size_param = 2 if len(df) < 100 else 20
df['cluster_refined'] = merge_small_clusters(X_weighted, df['cluster_initial'].values, min_size=min_size_param)

# Reassign cluster IDs to be consecutive
final_map = {old: new for new, old in enumerate(sorted(df['cluster_refined'].unique()))}
df['final_cluster'] = df['cluster_refined'].map(final_map)

# 5. Optimal Ordering (Imputation Order)
Z = linkage(X_weighted, method='ward')
sorted_indices = leaves_list(Z)
df_sorted = df.iloc[sorted_indices].reset_index(drop=True)
df_sorted['imputation_order'] = range(1, len(df_sorted) + 1)

print(f"Clustering completed: {df_sorted['final_cluster'].nunique()} clusters.")

# 6. Dimensionality Reduction (t-SNE)
n_samples = len(df_sorted)
perp = min(5, n_samples - 1) if n_samples < 50 else 30
tsne = TSNE(n_components=2, random_state=42, perplexity=perp, init='pca', learning_rate='auto')

X_tsne = tsne.fit_transform(X_weighted[sorted_indices])
df_sorted['tSNE_1'] = X_tsne[:, 0]
df_sorted['tSNE_2'] = X_tsne[:, 1]

# 7. Visualization (t-SNE only)
fig, ax = plt.subplots(1, 1, figsize=(8, 7), dpi=150)
sns.set_style("white")

num_clusters = df_sorted['final_cluster'].nunique()
palette = sns.color_palette("tab20", n_colors=max(1, num_clusters))

def plot_embedding(ax, x_col, y_col, title):
    sns.scatterplot(
        data=df_sorted, x=x_col, y=y_col, 
        hue='final_cluster', palette=palette, 
        s=150, alpha=0.9, edgecolor='none', ax=ax
    )
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(x_col.replace("_", " "), fontsize=12)
    ax.set_ylabel(y_col.replace("_", " "), fontsize=12)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', ncol=1, frameon=False, title="Cluster ID")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1.5)

plot_embedding(ax, 'tSNE_1', 'tSNE_2', 't-SNE Visualization')

plt.tight_layout()
figure_png = "cluster_without_predefined_constraints_tsne.png"
figure_pdf = "cluster_without_predefined_constraints_tsne.pdf"
fig.savefig(figure_png, dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig(figure_pdf, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Figure saved: {figure_png}")
print(f"Figure saved: {figure_pdf}")

# 8. Save Results (Including Log Coverage)
output_cols = ['imputation_order', 'cell_index', 'cell_name', 'final_cluster', 'cell_methylation_rate', 'total_observed_cpg', 'coverage_log']
output_filename = "sorted_cells_log_scaled.csv"
df_sorted[output_cols].to_csv(output_filename, index=False)

print(f"CSV saved: {output_filename}")
print(df_sorted[['imputation_order', 'cell_index', 'cell_methylation_rate', 'coverage_log']].head())