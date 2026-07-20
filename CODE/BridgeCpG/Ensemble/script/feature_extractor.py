"""Neighborhood feature extraction for the Fig. 8 ensemble analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

WINDOW_SIZE = 10

def custom_find_peaks(data, height=None, distance=1):
    if len(data) < 3:
        return np.array([], dtype=int), {'peak_heights': np.array([])}
    
    peaks = []
    for i in range(1, len(data)-1):
        if data[i] > data[i-1] and data[i] > data[i+1]:
            peaks.append(i)
    peaks = np.array(peaks, dtype=int)
    
    if height is not None:
        peak_heights = data[peaks]
        valid_mask = peak_heights >= height
        peaks = peaks[valid_mask]
    
    if distance > 1 and len(peaks) > 1:
        filtered_peaks = [peaks[0]]
        for peak in peaks[1:]:
            if peak - filtered_peaks[-1] >= distance:
                filtered_peaks.append(peak)
        peaks = np.array(filtered_peaks, dtype=int)
    
    return peaks, {'peak_heights': data[peaks] if len(peaks) > 0 else np.array([])}

class AdvancedNeighborhoodFeatureExtractor:
    def __init__(self, window_size=WINDOW_SIZE):
        self.window_size = window_size
    
    def extract_features_for_dataset(self, data_dict):
        graph_data = data_dict["graph"]
        mamba_data = data_dict["mamba"]
        transformer_data = data_dict["transformer"]
        true_data = data_dict["true"]
        
        n_rows, n_cols = graph_data.shape
        all_features = []
        
        for col_idx in range(n_cols):
            graph_col = graph_data.iloc[:, col_idx].values
            mamba_col = mamba_data.iloc[:, col_idx].values
            transformer_col = transformer_data.iloc[:, col_idx].values
            true_col = true_data.iloc[:, col_idx].values
            
            col_features = self._extract_features_for_column(
                graph_col, mamba_col, transformer_col, true_col, col_idx
            )
            all_features.extend(col_features)
        
        features_df = pd.DataFrame(all_features)
        return features_df
    
    def _extract_features_for_column(self, graph_col, mamba_col, transformer_col, true_col, col_idx):
        features = []
        n_positions = len(graph_col)
        
        for pos in range(n_positions):
            if (np.isnan(graph_col[pos]) or np.isnan(mamba_col[pos]) or 
                np.isnan(transformer_col[pos]) or np.isnan(true_col[pos])):
                continue
            
            graph_val = graph_col[pos]
            mamba_val = mamba_col[pos]
            transformer_val = transformer_col[pos]
            
            weighted_avg = 0.25 * graph_val + 0.4 * mamba_val + 0.35 * transformer_val
            max_val = max(graph_val, mamba_val, transformer_val)
            min_val = min(graph_val, mamba_val, transformer_val)
            range_val = max_val - min_val
            std_val = np.std([graph_val, mamba_val, transformer_val])
            
            base_features = {
                'graph_current': graph_val,
                'mamba_current': mamba_val,
                'transformer_current': transformer_val,
                'weighted_avg': weighted_avg,  
                'max_score': max_val,          
                'min_score': min_val,          
                'score_range': range_val,      
                'score_std': std_val,          
                'avg_score': (graph_val + mamba_val + transformer_val) / 3,  
                'y_true': int(true_col[pos]),
                'col_idx': col_idx,
                'pos_idx': pos
            }
            
            start_idx = max(0, pos - self.window_size)
            end_idx = min(n_positions, pos + self.window_size + 1)
            
            graph_window = graph_col[start_idx:end_idx]
            mamba_window = mamba_col[start_idx:end_idx]
            transformer_window = transformer_col[start_idx:end_idx]
            
            graph_valid = graph_window[~np.isnan(graph_window)]
            mamba_valid = mamba_window[~np.isnan(mamba_window)]
            transformer_valid = transformer_window[~np.isnan(transformer_window)]
            
            advanced_features = {}
            advanced_features.update(self._extract_statistical_features(graph_valid, 'graph'))
            advanced_features.update(self._extract_statistical_features(mamba_valid, 'mamba'))
            advanced_features.update(self._extract_statistical_features(transformer_valid, 'transformer'))
            advanced_features.update(self._extract_shape_features(graph_valid, 'graph'))
            advanced_features.update(self._extract_shape_features(mamba_valid, 'mamba'))
            advanced_features.update(self._extract_shape_features(transformer_valid, 'transformer'))
            advanced_features.update(self._extract_positional_features(
                graph_col, mamba_col, transformer_col, pos, start_idx, end_idx
            ))
            if len(graph_valid) > 2 and len(mamba_valid) > 2:
                advanced_features.update(self._extract_correlation_features(
                    graph_valid, mamba_valid, transformer_valid
                ))
            
            all_sample_features = {**base_features, **advanced_features}
            features.append(all_sample_features)
        
        return features
    
    def _extract_statistical_features(self, data, prefix):
        features = {}
        if len(data) > 0:
            features.update({
                f'{prefix}_neighbor_mean': np.mean(data),
                f'{prefix}_neighbor_std': np.std(data) if len(data) > 1 else 0,
                f'{prefix}_neighbor_median': np.median(data),
                f'{prefix}_neighbor_range': np.ptp(data) if len(data) > 1 else 0,
                f'{prefix}_neighbor_iqr': stats.iqr(data) if len(data) > 4 else 0,
                f'{prefix}_neighbor_skew': stats.skew(data) if len(data) > 2 else 0,
                f'{prefix}_neighbor_kurtosis': stats.kurtosis(data) if len(data) > 3 else 0,
                f'{prefix}_neighbor_valid_count': len(data)
            })
            quantiles = np.quantile(data, [0.1, 0.25, 0.75, 0.9])
            for i, q in enumerate([10, 25, 75, 90]):
                features[f'{prefix}_neighbor_q{q}'] = quantiles[i]
        else:
            for stat in ['mean', 'std', 'median', 'range', 'iqr', 'skew', 'kurtosis', 'valid_count']:
                features[f'{prefix}_neighbor_{stat}'] = 0
            for q in [10, 25, 75, 90]:
                features[f'{prefix}_neighbor_q{q}'] = 0
        return features
    
    def _extract_shape_features(self, data, prefix):
        features = {}
        if len(data) > 2:
            try:
                x = np.arange(len(data))
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, data)
                features.update({
                    f'{prefix}_neighbor_trend_slope': slope,
                    f'{prefix}_neighbor_trend_r2': r_value**2,
                    f'{prefix}_neighbor_trend_pvalue': p_value
                })
                if len(data) > 4:
                    peaks, properties = custom_find_peaks(data, height=np.mean(data), distance=2)
                    features[f'{prefix}_neighbor_peak_count'] = len(peaks)
                    features[f'{prefix}_neighbor_peak_mean_height'] = np.mean(properties['peak_heights']) if len(peaks) > 0 else 0
                else:
                    features[f'{prefix}_neighbor_peak_count'] = 0
                    features[f'{prefix}_neighbor_peak_mean_height'] = 0
            except:
                features.update({
                    f'{prefix}_neighbor_trend_slope': 0,
                    f'{prefix}_neighbor_trend_r2': 0,
                    f'{prefix}_neighbor_trend_pvalue': 1,
                    f'{prefix}_neighbor_peak_count': 0,
                    f'{prefix}_neighbor_peak_mean_height': 0
                })
        else:
            features.update({
                f'{prefix}_neighbor_trend_slope': 0,
                f'{prefix}_neighbor_trend_r2': 0,
                f'{prefix}_neighbor_trend_pvalue': 1,
                f'{prefix}_neighbor_peak_count': 0,
                f'{prefix}_neighbor_peak_mean_height': 0
            })
        return features
    
    def _extract_positional_features(self, graph_col, mamba_col, transformer_col, pos, start_idx, end_idx):
        features = {}
        window_center = (start_idx + end_idx - 1) / 2
        relative_pos = pos - window_center
        features['position_relative_to_center'] = relative_pos
        features['distance_to_start'] = pos - start_idx
        features['distance_to_end'] = end_idx - 1 - pos
        graph_window = graph_col[start_idx:end_idx]
        mamba_window = mamba_col[start_idx:end_idx]
        transformer_window = transformer_col[start_idx:end_idx]
        graph_valid = graph_window[~np.isnan(graph_window)]
        mamba_valid = mamba_window[~np.isnan(mamba_window)]
        transformer_valid = transformer_window[~np.isnan(transformer_window)]
        if len(graph_valid) > 0:
            current_graph = graph_col[pos]
            features['graph_current_rank'] = np.mean(current_graph > graph_valid) if len(graph_valid) > 0 else 0.5
            features['graph_current_zscore'] = (current_graph - np.mean(graph_valid)) / (np.std(graph_valid) + 1e-8)
        if len(mamba_valid) > 0:
            current_mamba = mamba_col[pos]
            features['mamba_current_rank'] = np.mean(current_mamba > mamba_valid) if len(mamba_valid) > 0 else 0.5
            features['mamba_current_zscore'] = (current_mamba - np.mean(mamba_valid)) / (np.std(mamba_valid) + 1e-8)
        if len(transformer_valid) > 0:
            current_transformer = transformer_col[pos]
            features['transformer_current_rank'] = np.mean(current_transformer > transformer_valid) if len(transformer_valid) > 0 else 0.5
            features['transformer_current_zscore'] = (current_transformer - np.mean(transformer_valid)) / (np.std(transformer_valid) + 1e-8)
        return features
    
    def _extract_correlation_features(self, graph_data, mamba_data, transformer_data):
        features = {}
        if len(graph_data) == len(mamba_data):
            corr_graph_mamba = np.corrcoef(graph_data, mamba_data)[0, 1] if len(graph_data) > 1 else 0
            features['corr_graph_mamba'] = 0 if np.isnan(corr_graph_mamba) else corr_graph_mamba
        else:
            features['corr_graph_mamba'] = 0
        if len(graph_data) == len(transformer_data):
            corr_graph_transformer = np.corrcoef(graph_data, transformer_data)[0, 1] if len(graph_data) > 1 else 0
            features['corr_graph_transformer'] = 0 if np.isnan(corr_graph_transformer) else corr_graph_transformer
        else:
            features['corr_graph_transformer'] = 0
        if len(mamba_data) == len(transformer_data):
            corr_mamba_transformer = np.corrcoef(mamba_data, transformer_data)[0, 1] if len(mamba_data) > 1 else 0
            features['corr_mamba_transformer'] = 0 if np.isnan(corr_mamba_transformer) else corr_mamba_transformer
        else:
            features['corr_mamba_transformer'] = 0
        return features

def load_clean_data(file_paths, dataset_name):
    data_dict = {}
    for model_name, path in file_paths.items():
        try:
            df = pd.read_csv(path, header=0, index_col=0)
            df = df.iloc[1:, 1:]
            data_cols = df.columns
            if not data_cols.tolist():
                raise ValueError(f"{model_name} has no data columns after skipping first row/column")
            df_data = df.astype(np.float32)
            if df_data.columns.dtype == 'object' and isinstance(df_data.columns[0], tuple):
                df_data.columns = [''.join(col) for col in df_data.columns]
            data_dict[model_name] = df_data
        except Exception as e:
            raise RuntimeError(f"Failed to load {model_name} for {dataset_name}: {str(e)}")
    all_shapes = [df.shape for df in data_dict.values()]
    if len(set(all_shapes)) != 1:
        raise ValueError(f"All data shapes must be consistent for {dataset_name}! Actual: {all_shapes}")
    true_vals = data_dict["true"].dropna().values
    if not np.all(np.isin(true_vals, [0.0, 1.0])):
        raise ValueError(f"y_true contains non-binary values for {dataset_name}: {np.unique(true_vals)}")
    return data_dict

def prepare_samples(data_dict, dataset_name):
    extractor = AdvancedNeighborhoodFeatureExtractor(window_size=WINDOW_SIZE)
    samples_df = extractor.extract_features_for_dataset(data_dict)
    return samples_df
