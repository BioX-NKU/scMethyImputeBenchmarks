"""Train the LightGBM ensemble classifier on Part 2.

The ensemble decision threshold is calibrated exclusively from Part 2
out-of-fold predictions and saved with the trained model. Evaluation reuses
this frozen threshold and never optimizes it on Part 1/Test.
"""

from __future__ import annotations

import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, GroupKFold
try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:
    StratifiedGroupKFold = None

from feature_extractor import load_clean_data, prepare_samples


file_paths_train = {
    "graph": os.path.join("processed_data", "graphCpG_part2.csv"),
    "mamba": os.path.join("processed_data", "mamba_y_pred_prob_part2.csv"),
    "transformer": os.path.join(
        "processed_data", "transformer_y_pred_prob_part2.csv"
    ),
    "true": os.path.join("processed_data", "transformer_y_true_part2.csv"),
}

output_dir = "./meta_results"
os.makedirs(output_dir, exist_ok=True)

ENSEMBLE_PARAMS = {
    "boosting_type": "gbdt",
    "n_estimators": 300,
    "learning_rate": 0.025,
    "max_depth": 8,
    "num_leaves": 40,
    "min_child_samples": 25,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "reg_alpha": 0.2,
    "reg_lambda": 0.2,
    "random_state": 42,
    "verbose": 50,
    "n_jobs": -1,
}

THRESHOLD = 0.5

def _fit_lgbm_classifier(clf, X, y):
    """Fit LightGBM while remaining compatible with old/new LightGBM APIs."""
    try:
        clf.fit(X, y, verbose=50)
    except TypeError:
        clf.fit(X, y)
    return clf


def _build_leakage_safe_cv_splits(X, y, groups, n_splits=5):
    """Create out-of-fold splits using columns as groups whenever possible."""
    y = np.asarray(y, dtype=int)
    groups = np.asarray(groups)
    class_counts = np.bincount(y, minlength=2)
    if np.count_nonzero(class_counts) < 2:
        return [], "single-class (fixed threshold)"

    max_by_class = int(class_counts[class_counts > 0].min())
    unique_groups = np.unique(groups)
    requested_splits = max(2, min(int(n_splits), max_by_class))

    if StratifiedGroupKFold is not None and len(unique_groups) >= requested_splits:
        try:
            try:
                splitter = StratifiedGroupKFold(
                    n_splits=requested_splits, shuffle=True, random_state=42
                )
            except TypeError:
                splitter = StratifiedGroupKFold(n_splits=requested_splits)
            splits = list(splitter.split(X, y, groups))
            if all(np.unique(y[train_idx]).size == 2 for train_idx, _ in splits):
                return splits, f"StratifiedGroupKFold({requested_splits}) by col_idx"
        except ValueError:
            pass

    group_splits = min(requested_splits, len(unique_groups))
    if group_splits >= 2:
        try:
            splits = list(GroupKFold(n_splits=group_splits).split(X, y, groups))
            if all(np.unique(y[train_idx]).size == 2 for train_idx, _ in splits):
                return splits, f"GroupKFold({group_splits}) by col_idx"
        except ValueError:
            pass

    sample_splits = min(requested_splits, max_by_class)
    if sample_splits >= 2:
        splitter = StratifiedKFold(
            n_splits=sample_splits, shuffle=True, random_state=42
        )
        return list(splitter.split(X, y)), f"StratifiedKFold({sample_splits}) fallback"

    return [], "no valid CV split (fixed threshold)"


def _score_at_threshold(y_true, y_score, threshold):
    """Return Accuracy, F1 and MCC for one fixed threshold."""
    y_true = np.asarray(y_true, dtype=np.int8)
    y_score = np.asarray(y_score, dtype=np.float64)
    y_pred = (y_score >= float(threshold)).astype(np.int8)

    tp = float(np.sum((y_true == 1) & (y_pred == 1)))
    tn = float(np.sum((y_true == 0) & (y_pred == 0)))
    fp = float(np.sum((y_true == 0) & (y_pred == 1)))
    fn = float(np.sum((y_true == 1) & (y_pred == 0)))

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0.0
    f1_denom = 2 * tp + fp + fn
    f1 = (2 * tp) / f1_denom if f1_denom else 0.0
    mcc_denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn - fp * fn) / mcc_denom) if mcc_denom else 0.0
    return float(accuracy), float(f1), float(mcc)


def find_optimal_threshold(y_true, y_score):
    """
    Calibrate a decision threshold on training/calibration predictions only.

    The search is exact over score breakpoints (plus 0.05/0.50/0.95), rather
    than a coarse 61-point grid.  Among thresholds whose accuracy is within
    0.5 percentage points of the best accuracy, it selects the highest F1;
    MCC and proximity to 0.5 are deterministic tie-breakers.

    IMPORTANT: calculate_metrics() never calls this function.  The final test
    set therefore cannot influence the chosen threshold.
    """
    y_true = np.asarray(y_true, dtype=np.int8)
    y_score = np.asarray(y_score, dtype=np.float64)
    valid_mask = np.isfinite(y_score) & np.isin(y_true, [0, 1])
    y_true = y_true[valid_mask]
    y_score = y_score[valid_mask]

    if y_true.size == 0:
        return float(THRESHOLD), 0.0, 0.0
    if np.unique(y_true).size < 2:
        accuracy, f1, _ = _score_at_threshold(y_true, y_score, THRESHOLD)
        return float(THRESHOLD), f1, accuracy

    score_breakpoints = np.unique(y_score[(y_score >= 0.05) & (y_score <= 0.95)])
    thresholds = np.unique(
        np.concatenate((score_breakpoints, np.array([0.05, THRESHOLD, 0.95])))
    )

    order = np.argsort(y_score, kind="mergesort")
    sorted_scores = y_score[order]
    sorted_y = y_true[order]
    suffix_positive = np.r_[np.cumsum(sorted_y[::-1], dtype=np.int64)[::-1], 0]

    first_positive_idx = np.searchsorted(sorted_scores, thresholds, side="left")
    tp = suffix_positive[first_positive_idx].astype(np.float64)
    predicted_positive = (len(y_true) - first_positive_idx).astype(np.float64)
    fp = predicted_positive - tp
    total_positive = float(np.sum(y_true == 1))
    total_negative = float(np.sum(y_true == 0))
    fn = total_positive - tp
    tn = total_negative - fp

    accuracy = (tp + tn) / len(y_true)
    f1_denom = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, f1_denom, out=np.zeros_like(tp), where=f1_denom > 0)
    mcc_denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = np.divide(
        tp * tn - fp * fn,
        mcc_denom,
        out=np.zeros_like(tp),
        where=mcc_denom > 0,
    )

    max_accuracy = float(np.max(accuracy))
    eligible = np.flatnonzero(accuracy >= max_accuracy - 0.005)
    best_idx = max(
        eligible.tolist(),
        key=lambda i: (
            float(f1[i]),
            float(accuracy[i]),
            float(mcc[i]),
            -abs(float(thresholds[i]) - float(THRESHOLD)),
        ),
    )
    return (
        float(thresholds[best_idx]),
        float(f1[best_idx]),
        float(accuracy[best_idx]),
    )


def train_ensemble_classifier(train_samples_df):
    exclude_cols = ['col_idx', 'pos_idx', 'y_true']
    feature_cols = [col for col in train_samples_df.columns if col not in exclude_cols]
    X_train = train_samples_df[feature_cols].values
    y_train = train_samples_df["y_true"].values.astype(int)
    if len(X_train) == 0:
        raise ValueError("No valid training samples, cannot train classifier")

    groups = (
        train_samples_df["col_idx"].values
        if "col_idx" in train_samples_df.columns
        else np.arange(len(train_samples_df))
    )
    cv_splits, cv_method = _build_leakage_safe_cv_splits(
        X_train, y_train, groups, n_splits=5
    )

    decision_threshold = float(THRESHOLD)
    calibration_f1 = np.nan
    calibration_accuracy = np.nan
    calibration_mcc = np.nan

    if cv_splits:
        oof_proba = np.full(len(y_train), np.nan, dtype=np.float64)
        for fold_idx, (fit_idx, valid_idx) in enumerate(cv_splits, start=1):
            print(
                f"Threshold calibration fold {fold_idx}/{len(cv_splits)} "
                f"({cv_method})"
            )
            fold_clf = LGBMClassifier(**ENSEMBLE_PARAMS)
            _fit_lgbm_classifier(fold_clf, X_train[fit_idx], y_train[fit_idx])
            oof_proba[valid_idx] = fold_clf.predict_proba(X_train[valid_idx])[:, 1]

        if np.all(np.isfinite(oof_proba)):
            decision_threshold, calibration_f1, calibration_accuracy = (
                find_optimal_threshold(y_train, oof_proba)
            )
            _, _, calibration_mcc = _score_at_threshold(
                y_train, oof_proba, decision_threshold
            )
        else:
            print(
                "Warning: incomplete OOF predictions; using the fixed 0.5 "
                "threshold instead of consulting any test labels."
            )
    else:
        print(
            "Warning: unable to create leakage-safe OOF folds; using fixed "
            "threshold 0.5."
        )

    print(
        "Leakage-safe ensemble threshold (calibrated on part2 OOF only): "
        f"{decision_threshold:.6f}; F1={calibration_f1:.4f}, "
        f"Accuracy={calibration_accuracy:.4f}, MCC={calibration_mcc:.4f}"
    )

    clf = LGBMClassifier(**ENSEMBLE_PARAMS)
    _fit_lgbm_classifier(clf, X_train, y_train)
    feature_info = {
        'feature_columns': feature_cols,
        'classifier': clf,
        'feature_importance': dict(zip(feature_cols, clf.feature_importances_)),
        'decision_threshold': float(decision_threshold),
        'threshold_calibration': {
            'source': 'part2 training out-of-fold predictions only',
            'cv_method': cv_method,
            'folds': len(cv_splits),
            'f1': float(calibration_f1) if np.isfinite(calibration_f1) else None,
            'accuracy': float(calibration_accuracy) if np.isfinite(calibration_accuracy) else None,
            'mcc': float(calibration_mcc) if np.isfinite(calibration_mcc) else None,
        },
    }
    clf_save_path = f"{output_dir}/ensemble_classifier_balanced_f1_acc.pkl"
    joblib.dump(feature_info, clf_save_path)
    return feature_info

def compute_single_model_metrics(y_true, y_pred_binary, y_score):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    metrics = {
        "acc": accuracy_score(y_true, y_pred_binary),
        "mcc": matthews_corrcoef(y_true, y_pred_binary),
        "tpr": tpr,
        "tnr": tnr,
        "f1": f1_score(y_true, y_pred_binary),
        "auc": roc_auc_score(y_true, y_score)
    }
    return metrics

def calculate_metrics(samples_df, ensemble_clf_info=None, dataset_name=""):
    metrics_dict = {}
    y_true = samples_df["y_true"].values
    for model_name in ["graph", "mamba", "transformer", "avg", "weighted_avg"]:
        if model_name == "avg":
            y_score = samples_df["avg_score"].values
        elif model_name == "weighted_avg":
            y_score = samples_df["weighted_avg"].values
        else:
            y_score = samples_df[f"{model_name}_current"].values
        y_pred_binary = (y_score >= THRESHOLD).astype(int)
        metrics = compute_single_model_metrics(y_true, y_pred_binary, y_score)
        metrics_dict[model_name] = metrics

    if ensemble_clf_info is not None:
        feature_cols = ensemble_clf_info['feature_columns']
        clf = ensemble_clf_info['classifier']
        X_data = samples_df[feature_cols].values
        y_pred_ensemble_proba = clf.predict_proba(X_data)[:, 1]

        # Critical leakage fix: use the threshold saved during part2-only OOF
        # calibration.  y_true from this dataset is used only for reporting.
        ensemble_threshold = float(
            ensemble_clf_info.get('decision_threshold', THRESHOLD)
        )
        y_pred_ensemble_binary = (
            y_pred_ensemble_proba >= ensemble_threshold
        ).astype(int)
        metrics_ensemble = compute_single_model_metrics(
            y_true, y_pred_ensemble_binary, y_pred_ensemble_proba
        )
        metrics_dict["ensemble"] = metrics_ensemble
        samples_df["ensemble_score"] = y_pred_ensemble_proba
        samples_df["ensemble_pred"] = y_pred_ensemble_binary
        samples_df["ensemble_best_thresh"] = ensemble_threshold
        print(
            f"{dataset_name}: ensemble metrics use fixed training-calibrated "
            f"threshold={ensemble_threshold:.6f}; no threshold search on this set."
        )

    metrics_df = pd.DataFrame(metrics_dict).T
    metrics_df.columns = ["Accuracy", "MCC", "Sensitivity", "Specificity", "F1-Score","AUC"]
    metrics_df.index = [
        "GraphCpG", 
        "MambaCpG", 
        "CpGTransformer", 
        "Average of Three Models",
        "Weighted Average of Three Models",
        "Ensemble Classifier (Balanced F1 & Acc)"
    ]
    metrics_df = metrics_df.round(4)
    return metrics_df, samples_df

def plot_roc_comparison(samples_df, save_dir, dataset_name):
    plt.figure(figsize=(10, 8))
    model_config = {
        "graph": ("GraphCpG", "#1f77b4"),
        "mamba": ("MambaCpG", "#ff7f0e"),
        "transformer": ("CpGTransformer", "#2ca02c"),
        "avg": ("Average of Three Models", "#9467bd"),
        "weighted_avg": ("Weighted Average", "#8c564b"),
        "ensemble": ("Ensemble Classifier (Balanced F1 & Acc)", "#d62728")
    }
    for model_key, (model_label, color) in model_config.items():
        y_true = samples_df["y_true"].values
        if model_key == "ensemble":
            if "ensemble_score" in samples_df.columns:
                y_score = samples_df["ensemble_score"].values
            else:
                continue
        elif model_key == "avg":
            y_score = samples_df["avg_score"].values
        elif model_key == "weighted_avg":
            y_score = samples_df["weighted_avg"].values
        else:
            y_score = samples_df[f"{model_key}_current"].values
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc = roc_auc_score(y_true, y_score)
        plt.plot(fpr, tpr, color=color, lw=2,label=f"{model_label} (AUC = {auc:.4f})")
    plt.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random Guess")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (FPR)", fontsize=12)
    plt.ylabel("True Positive Rate (TPR)", fontsize=12)
    plt.title(f"ROC Curve Comparison - {dataset_name}", fontsize=14, fontweight="bold", pad=20)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    save_path = f"{save_dir}/roc_comparison_{dataset_name.lower().replace(' ', '_')}.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

def analyze_feature_importance(ensemble_clf_info, save_dir, top_n=30):
    feature_importance = ensemble_clf_info['feature_importance']
    feature_cols = ensemble_clf_info['feature_columns']
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': [feature_importance[col] for col in feature_cols]
    }).sort_values('importance', ascending=False)
    plt.figure(figsize=(12, 10))
    top_features = importance_df.head(top_n)
    colors = plt.cm.plasma(np.linspace(0, 1, len(top_features)))
    bars = plt.barh(range(len(top_features)), top_features['importance'], color=colors)
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Feature Importance Score', fontsize=12)
    plt.title(f'Top {top_n} Feature Importance (Balanced F1 & Acc)', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    save_path = f"{save_dir}/feature_importance_balanced.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    importance_path = f"{save_dir}/feature_importance_balanced.csv"
    importance_df.to_csv(importance_path, index=False)
    feature_categories = {
        'Enhanced Base Scores': ['graph_current', 'mamba_current', 'transformer_current', 
                                'weighted_avg', 'max_score', 'min_score', 'score_range', 'score_std'],
        'Statistical': ['mean', 'std', 'median', 'range', 'iqr', 'skew', 'kurtosis', 'q10', 'q25', 'q75', 'q90'],
        'Shape': ['trend', 'peak'],
        'Positional': ['position', 'distance', 'rank', 'zscore'],
        'Correlation': ['corr']
    }
    category_importance = {}
    for category, keywords in feature_categories.items():
        category_features = [col for col in feature_cols if any(keyword in col for keyword in keywords)]
        category_importance[category] = sum(feature_importance[col] for col in category_features if col in feature_importance)
    plt.figure(figsize=(10, 6))
    categories = list(category_importance.keys())
    importances = [category_importance[cat] for cat in categories]
    plt.bar(categories, importances, color=plt.cm.Set3(np.linspace(0, 1, len(categories))))
    plt.xlabel('Feature Category', fontsize=12)
    plt.ylabel('Total Importance Score', fontsize=12)
    plt.title('Feature Importance by Category', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45)
    plt.tight_layout()
    category_path = f"{save_dir}/feature_importance_by_category.png"
    plt.savefig(category_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("\nTop 10 most important features:")
    for i, (_, row) in enumerate(importance_df.head(10).iterrows()):
        print(f"   {i+1:2d}. {row['feature']}: {row['importance']:.4f}")
    print(f"\nFeature importance by category:")
    for category, importance in category_importance.items():
        print(f"   - {category}: {importance:.4f}")

def create_comparison_table(metrics_train, metrics_test):
    comparison_data = {}
    for model in metrics_train.index:
        comparison_data[f"{model} (part2 - Train)"] = metrics_train.loc[model]
    for model in metrics_test.index:
        comparison_data[f"{model} (part1 - Test)"] = metrics_test.loc[model]
    comparison_df = pd.DataFrame(comparison_data).T
    comparison_df = comparison_df.round(4)
    return comparison_df


def main() -> None:
    try:
        print("Training ensemble classifier on Part 2.")

        print("\n1. Loading training data")
        data_train = load_clean_data(file_paths_train, "part2 (Training)")

        print("\n2. Extracting neighborhood features")
        train_samples_df = prepare_samples(data_train, "part2 (Training)")

        print("\n3. Training LightGBM ensemble classifier")
        ensemble_clf_info = train_ensemble_classifier(train_samples_df)

        print("\n4. Calculating training metrics")
        metrics_train, train_samples_with_pred = calculate_metrics(
            train_samples_df,
            ensemble_clf_info,
            "part2 (Training)",
        )

        print("\nTraining Set Metrics (part2):")
        print(metrics_train.to_string())

        train_samples_with_pred.to_csv(
            f"{output_dir}/part2_training_detailed_results_balanced.csv",
            index=False,
        )
        metrics_train.to_csv(
            f"{output_dir}/part2_training_metrics_balanced.csv"
        )

        analyze_feature_importance(ensemble_clf_info, output_dir)
        plot_roc_comparison(
            train_samples_with_pred,
            output_dir,
            "part2 (Training)",
        )

        print("\nClassification training completed.")
        print(f"Training samples: {len(train_samples_df)}")
        print(
            f"Total features: "
            f"{len(ensemble_clf_info['feature_columns'])}"
        )
        print(f"Model and results saved to: {os.path.abspath(output_dir)}")

    except Exception as error:
        print(f"\nProcess interrupted: {error}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
