import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import euclidean_distances
from scipy.cluster.hierarchy import linkage, leaves_list
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore')

INFO_FILE_PATH = "hFC.csv"
OUTPUT_DIR = "methylation_clustering_results_hFC"

CLUSTER_THRESHOLD = 5
WEIGHT_RATE = 50
WEIGHT_COV = 1
MIN_CLUSTER_SIZE = 10
MAX_CLUSTER_SIZE = 100

def recursively_split_large_clusters(X_weighted, labels, max_cluster_size=MAX_CLUSTER_SIZE):
    if max_cluster_size < 1:
        raise ValueError("max_cluster_size must be at least 1")

    labels = np.asarray(labels)
    final_labels = np.full(len(labels), -1, dtype=int)
    next_label = 0

    def split_cluster(indices):
        nonlocal next_label

        if len(indices) <= max_cluster_size:
            final_labels[indices] = next_label
            next_label += 1
            return

        split_model = AgglomerativeClustering(
            n_clusters=2,
            linkage='ward'
        )
        split_labels = split_model.fit_predict(X_weighted[indices])

        for split_label in np.unique(split_labels):
            split_indices = indices[split_labels == split_label]
            split_cluster(split_indices)

    for label in np.unique(labels):
        cluster_indices = np.where(labels == label)[0]
        split_cluster(cluster_indices)

    return final_labels

def load_and_preprocess_metadata(info_file_path):
    print("Loading metadata file...")
    try:
        df = pd.read_csv(info_file_path)
    except FileNotFoundError:
        print(f"Error: File {info_file_path} not found")
        return None

    print(f"Data shape: {df.shape}")
    
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    
    meth_col = None
    for col in df.columns:
        if 'MethRatio' in col or 'methylation_rate' in col:
            meth_col = col
            break
    
    if meth_col:
        print(f"Using methylation rate column: {meth_col}")
        df['methylation_rate'] = df[meth_col]
    else:
        print("Error: No methylation rate related column found")
        return None
        
    total_cpg_col = None
    for col in df.columns:
        if 'totalCpGs' in col or 'total_observed_cpg' in col:
            total_cpg_col = col
            break
            
    if total_cpg_col:
        print(f"Using total CpG column: {total_cpg_col}")
        df['total_observed_cpg'] = df[total_cpg_col]
    else:
        cov_cols = [c for c in df.columns if 'Cov' in c]
        if cov_cols:
            print(f"Using coverage to estimate weight: {cov_cols[0]}")
            df['total_observed_cpg'] = df[cov_cols[0]] * 1000
        else:
            print("Warning: No CpG count information, weights will be uniform")
            df['total_observed_cpg'] = 1000

    if 'sample_id' not in df.columns:
        id_candidates = ['sampleName', 'cell_ID', 'cell_index']
        for c in id_candidates:
            if c in df.columns:
                df['sample_id'] = df[c]
                break
        if 'sample_id' not in df.columns:
            df['sample_id'] = df.index.astype(str)

    if 'cell_type' not in df.columns:
        print("Warning: No cell_type column found, will skip ground truth boxplot")
    else:
        df = df.dropna(subset=['cell_type'])

    df = df.dropna(subset=['methylation_rate', 'total_observed_cpg'])
    
    return df

def perform_clustering_and_viz(df, output_dir=OUTPUT_DIR, max_cluster_size=MAX_CLUSTER_SIZE):
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("Starting clustering analysis (renamed by methylation rate order)...")
    print("=" * 60)
    
    df['coverage_log'] = np.log10(df['total_observed_cpg'])
    
    X_raw_rate = df[['methylation_rate']].values
    X_log_cov = df[['coverage_log']].values
    X_combined = np.hstack([X_raw_rate, X_log_cov])
    
    weights = np.array([WEIGHT_RATE, WEIGHT_COV])
    X_weighted = X_combined * weights
    
    cluster_model = AgglomerativeClustering(
        distance_threshold=CLUSTER_THRESHOLD,
        n_clusters=None,
        linkage='ward'
    )
    raw_labels = cluster_model.fit_predict(X_weighted)
    
    df_temp = pd.DataFrame(X_weighted)
    df_temp['label'] = raw_labels
    
    min_size = MIN_CLUSTER_SIZE
    
    counts = df_temp['label'].value_counts()
    valid_labels = counts[counts >= min_size].index.values
    
    if len(valid_labels) > 1:
        valid_centroids = df_temp[df_temp['label'].isin(valid_labels)].groupby('label').mean().values
        valid_lbl_map = df_temp[df_temp['label'].isin(valid_labels)].groupby('label').mean().index.values
        
        refined_labels = raw_labels.copy()
        for label in np.unique(raw_labels):
            if counts[label] < min_size:
                indices = np.where(raw_labels == label)[0]
                for idx in indices:
                    sample = X_weighted[idx].reshape(1, -1)
                    dists = euclidean_distances(sample, valid_centroids)
                    refined_labels[idx] = valid_lbl_map[np.argmin(dists)]
        df['temp_cluster'] = refined_labels
    else:
        df['temp_cluster'] = raw_labels

    cluster_sizes_before_split = df['temp_cluster'].value_counts()
    oversized_clusters = cluster_sizes_before_split[cluster_sizes_before_split > max_cluster_size]

    if len(oversized_clusters) > 0:
        print(
            f"Recursively splitting {len(oversized_clusters)} cluster(s) larger than "
            f"{max_cluster_size} samples..."
        )
        df['temp_cluster'] = recursively_split_large_clusters(
            X_weighted,
            df['temp_cluster'].to_numpy(),
            max_cluster_size=max_cluster_size
        )

    max_final_cluster_size = int(df['temp_cluster'].value_counts().max())
    print(f"Maximum cluster size after recursive splitting: {max_final_cluster_size}")

    cluster_means = df.groupby('temp_cluster')['methylation_rate'].mean().sort_values(ascending=True)
    sorted_mapping = {old_id: new_id for new_id, old_id in enumerate(cluster_means.index)}
    df['final_cluster'] = df['temp_cluster'].map(sorted_mapping)
    df['cluster_name'] = df['final_cluster'].apply(lambda x: f"Cluster {x}")
    
    num_clusters = df['final_cluster'].nunique()
    print(f"Final number of clusters: {num_clusters}")
    print("Renamed by mean methylation rate (ascending order: Cluster 0 = lowest methylation)")

    print("Calculating Linkage and t-SNE...")
    
    Z = linkage(X_weighted, method='ward')
    sorted_indices = leaves_list(Z)
    
    df_sorted = df.iloc[sorted_indices].reset_index(drop=True)
    df_sorted['imputation_order'] = range(1, len(df_sorted) + 1)
    
    X_sorted_weighted = X_weighted[sorted_indices]
    
    n_samples = len(df_sorted)
    perp = min(5, n_samples - 1) if n_samples < 50 else 30
    print(f"Using Perplexity: {perp} (Total Samples: {n_samples})")
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=perp, init='pca', learning_rate='auto')
    X_tsne = tsne.fit_transform(X_sorted_weighted)
    
    df_sorted['tSNE_1'] = X_tsne[:, 0]
    df_sorted['tSNE_2'] = X_tsne[:, 1]

    sns.set_style("white")
    fig, ax = plt.subplots(1, 1, figsize=(8, 7), dpi=150)
    
    palette = sns.color_palette("tab20", n_colors=max(1, num_clusters))
    
    sns.scatterplot(
        data=df_sorted, 
        x='tSNE_1', 
        y='tSNE_2', 
        hue='final_cluster', 
        palette=palette, 
        s=150,
        alpha=0.9,
        edgecolor='none',
        ax=ax
    )
    
    ax.set_xlabel('t-SNE 1', fontsize=20)
    ax.set_ylabel('t-SNE 2', fontsize=20)
    
    ax.legend(
        bbox_to_anchor=(1.02, 1), 
        loc='upper left', 
        ncol=1, 
        frameon=False, 
        title="Cluster ID"
    )
    
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1.5)
    
    ax.set_title("")
    
    plt.tight_layout()
    tsne_save_path = os.path.join(output_dir, 'tsne_clusters.png')
    plt.savefig(tsne_save_path, dpi=300)
    plt.close()
    print(f"t-SNE Plot saved: {tsne_save_path}")

    def draw_sorted_boxplot(data, x_col, filename, title_prefix, color_seed=42):
        group_means = data.groupby(x_col)['methylation_rate'].mean().sort_values()
        sorted_groups = group_means.index.tolist()
        
        df_plot = data.copy()
        df_all = data.copy()
        df_all[x_col] = 'All'
        
        df_combined = pd.concat([df_plot, df_all], ignore_index=True)
        
        plot_order = sorted_groups + ['All']
        
        n_groups = len(sorted_groups)
        base_palette = plt.cm.tab20(np.linspace(0, 1, n_groups)) if n_groups <= 20 else plt.cm.nipy_spectral(np.linspace(0, 1, n_groups))
        final_palette = list(base_palette) + ['lightgray']
        
        plt.figure(figsize=(max(10, len(plot_order)*0.6), 8), dpi=150)
        
        sns.boxplot(
            data=df_combined,
            x=x_col,
            y='methylation_rate',
            order=plot_order,
            palette=final_palette,
            linewidth=1.5,
            flierprops=dict(marker='o', markerfacecolor='red', markersize=3, alpha=0.5)
        )
        
        plt.title('', fontsize=14)
        plt.xlabel('', fontsize=12)
        plt.ylabel('Methylation Rate', fontsize=20)
        
        current_labels = [label.get_text() for label in plt.gca().get_xticklabels()]
        new_labels = []
        for lbl in current_labels:
            if lbl.startswith('Cluster '):
                new_lbl = lbl.replace('Cluster ', '')
                new_labels.append(new_lbl)
            else:
                new_labels.append(lbl)
        plt.gca().set_xticklabels(new_labels, rotation=0, ha='center', fontsize=20)
        
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        save_path = os.path.join(output_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Boxplot saved: {save_path}")
        plt.close()

    draw_sorted_boxplot(
        df_sorted, 
        x_col='cluster_name', 
        filename='boxplot_by_cluster.png', 
        title_prefix=''
    )

    if 'cell_type' in df_sorted.columns:
        print("Plotting cell type boxplot...")
        draw_sorted_boxplot(
            df_sorted, 
            x_col='cell_type', 
            filename='boxplot_by_cell_type.png', 
            title_prefix=''
        )
    
    output_cols = ['sample_id', 'cluster_name', 'methylation_rate', 'total_observed_cpg',
                   'coverage_log', 'imputation_order', 'tSNE_1', 'tSNE_2']
    if 'cell_type' in df_sorted.columns:
        output_cols.append('cell_type')
        
    extra_cols = ['treatment', 'subgroup', 'tissue', 'age']
    for c in extra_cols:
        if c in df_sorted.columns:
            output_cols.append(c)
    
    output_cols = [col for col in output_cols if col in df_sorted.columns]
            
    df_sorted[output_cols].to_csv(os.path.join(output_dir, 'clustering_result_table.csv'), index=False)
    print(f"Result table saved.")

    print("\nCluster Summary Statistics (Mean Methylation Rate):")
    summary = df_sorted.groupby('cluster_name')['methylation_rate'].agg(['count', 'mean', 'std']).sort_values('mean')
    print(summary)

    return df_sorted

if __name__ == "__main__":
    df = load_and_preprocess_metadata(INFO_FILE_PATH)
    
    if df is not None:
        perform_clustering_and_viz(df)
        print("All tasks completed.")