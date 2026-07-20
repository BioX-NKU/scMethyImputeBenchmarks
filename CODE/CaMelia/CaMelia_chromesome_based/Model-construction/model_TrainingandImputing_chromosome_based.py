# -*- coding: utf-8 -*-
"""
Chromosome-based CaMelia training and evaluation.

Usage:
    python model_TrainingandImputing_chromosome_based.py DATA_PATH INPUT_NAME GPU [TEST_CHROMOSOME]

Example:
    python model_TrainingandImputing_chromosome_based.py ./data dataset GPU chr10

The default held-out chromosome is chr10.
"""
from __future__ import division

from sys import argv
import os
import re
import time
import warnings

import catboost as cb
import matplotlib as mpl
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import f1_score

mpl.use("Agg")
warnings.filterwarnings("ignore")


def reduce_mem(df):
    """Reduce DataFrame memory usage by downcasting numeric data types."""
    starttime = time.time()
    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage().sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtypes
        if col_type not in numerics:
            continue

        c_min = df[col].min()
        c_max = df[col].max()
        if pd.isnull(c_min) or pd.isnull(c_max):
            continue

        if str(col_type)[:3] == "int":
            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
            else:
                df[col] = df[col].astype(np.int64)
        else:
            if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                df[col] = df[col].astype(np.float16)
            elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                df[col] = df[col].astype(np.float32)
            else:
                df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    reduction_percent = 100 * (start_mem - end_mem) / start_mem if start_mem else 0
    time_spent = (time.time() - starttime) / 60
    print(
        "-- Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction), "
        "time spent: {:2.2f} min".format(
            end_mem, reduction_percent, time_spent
        )
    )
    return df


def normalize_chromosome(value):
    """Normalize chromosome labels such as 10, 10.0, chr10 and Chr10 to chr10."""
    value = str(value).strip().lower()

    if value.endswith(".0") and value[:-2].isdigit():
        value = value[:-2]

    if value.startswith("chromosome"):
        value = value.replace("chromosome", "", 1).strip()

    if not value.startswith("chr"):
        value = "chr" + value

    if value == "chrmt":
        value = "chrm"

    return value


def is_autosome(chromosome):
    """Return True only for labels of the form chr1, chr2, ..., chrN."""
    return re.fullmatch(r"chr\d+", chromosome) is not None


def calculate_binary_metrics(y_true, probabilities, threshold=0.5):
    """Calculate ACC, AUC, specificity, sensitivity, MCC and F1."""
    y_true = np.asarray(y_true, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=float)
    y_pred = (probabilities >= threshold).astype(np.int64)

    acc_value = metrics.accuracy_score(y_true, y_pred)
    f1_value = f1_score(y_true, y_pred, zero_division=0)
    mcc_value = metrics.matthews_corrcoef(y_true, y_pred)

    if np.unique(y_true).size >= 2:
        auc_value = metrics.roc_auc_score(y_true, probabilities)
    else:
        auc_value = np.nan

    tn, fp, fn, tp = metrics.confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan

    return {
        "SP": specificity,
        "SE": sensitivity,
        "ACC": acc_value,
        "AUC": auc_value,
        "MCC": mcc_value,
        "F1": f1_value,
    }


def build_catboost_model(seed, task_type):
    return cb.CatBoostClassifier(
        random_state=seed,
        learning_rate=0.1,
        max_depth=7,
        verbose=1000,
        eval_metric="AUC",
        task_type=task_type,
    )


if __name__ == "__main__":
    if len(argv) < 4:
        raise ValueError(
            "Usage: python model_TrainingandImputing_chromosome_based.py "
            "DATA_PATH INPUT_NAME GPU_OR_CPU [TEST_CHROMOSOME]"
        )

    DataPath = r"%s" % argv[1]
    InputDataName = "%s" % argv[2]  # Retained for compatibility with the original script.
    t_t = "%s" % argv[3]
    test_chromosome = normalize_chromosome(argv[4] if len(argv) > 4 else "chr10")

    gse = DataPath
    seed = 514

    meragefiledir_out = r"%s/Available_Train_dataset/region10" % gse
    meragefiledir_impu = r"%s/Available_Imputation_dataset/region10" % gse

    filenames_out = sorted(
        filename
        for filename in os.listdir(meragefiledir_out)
        if os.path.isfile(os.path.join(meragefiledir_out, filename))
    )

    file_dir_png = r"%s/model_feature_importance" % gse
    file_dir_pre = r"%s/imputation_data" % gse
    file_dir_model = r"%s/model_save" % gse

    os.makedirs(file_dir_png, exist_ok=True)
    os.makedirs(file_dir_pre, exist_ok=True)
    os.makedirs(file_dir_model, exist_ok=True)

    result_rows = []

    for filename in filenames_out:
        train_path = os.path.join(meragefiledir_out, filename)
        impu_path = os.path.join(meragefiledir_impu, filename)

        if not os.path.exists(impu_path):
            print("Warning: matching imputation file not found, skipped: %s" % impu_path)
            continue

        dataset_train = pd.read_csv(train_path, header=0, sep="\t")
        dataset_train = reduce_mem(dataset_train)
        dataset_train = dataset_train.dropna(axis=1, how="all")
        dataset_train = dataset_train.dropna(axis=0, how="any")

        dataset_impu = pd.read_csv(impu_path, header=0, sep="\t")
        dataset_impu = reduce_mem(dataset_impu)
        dataset_impu = dataset_impu.dropna(axis=1, how="all")
        dataset_impu = dataset_impu.dropna(axis=0, how="any")

        if dataset_train.shape[1] < 4:
            raise ValueError(
                "%s has fewer than four columns. Expected chromosome, position, "
                "target and model features." % filename
            )

        if dataset_impu.shape[1] < 3:
            raise ValueError(
                "%s has fewer than three imputation columns. Expected chromosome, "
                "position and model features." % filename
            )

        train_chromosomes = dataset_train.iloc[:, 0].map(normalize_chromosome)
        autosome_mask = train_chromosomes.map(is_autosome).to_numpy()
        test_mask = (train_chromosomes == test_chromosome).to_numpy()
        model_train_mask = autosome_mask & (~test_mask)

        if model_train_mask.sum() == 0:
            raise ValueError(
                "%s contains no autosomal training rows outside %s."
                % (filename, test_chromosome)
            )

        if test_mask.sum() == 0:
            raise ValueError(
                "%s contains no rows from held-out chromosome %s."
                % (filename, test_chromosome)
            )

        dataset_values = dataset_train.values
        X = dataset_values[:, 3:]
        Y = np.int64(dataset_values[:, 2] >= 0.5)

        X_train = X[model_train_mask, :]
        y_train = Y[model_train_mask]
        X_test = X[test_mask, :]
        y_test = Y[test_mask]

        print(
            "%s: chromosome hold-out split completed "
            "(train rows=%d, %s test rows=%d)."
            % (
                filename.split(".")[0],
                len(y_train),
                test_chromosome,
                len(y_test),
            )
        )

        model = build_catboost_model(seed=seed, task_type=t_t)
        model.fit(X_train, y_train)

        test_probabilities = model.predict_proba(X_test)[:, 1]
        metric_values = calculate_binary_metrics(
            y_true=y_test,
            probabilities=test_probabilities,
            threshold=0.5,
        )

        result_rows.append(
            {
                "cell": filename.split(".")[0],
                "test_chromosome": test_chromosome,
                "train_rows": len(y_train),
                "test_rows": len(y_test),
                "SP": round(metric_values["SP"], 4)
                if not np.isnan(metric_values["SP"])
                else np.nan,
                "SE": round(metric_values["SE"], 4)
                if not np.isnan(metric_values["SE"])
                else np.nan,
                "ACC": round(metric_values["ACC"], 4),
                "AUC": round(metric_values["AUC"], 4)
                if not np.isnan(metric_values["AUC"])
                else np.nan,
                "MCC": round(metric_values["MCC"], 4),
                "F1": round(metric_values["F1"], 4),
            }
        )

        model_path = os.path.join(
            file_dir_model, "%s.model" % filename.split(".")[0]
        )
        model.save_model(model_path)
        print("%s: Model Training Completed!" % filename.split(".")[0])
        print("----------------")
        print("%s: Imputation Process Started!" % filename.split(".")[0])

        impu_chromosomes = dataset_impu.iloc[:, 0].map(normalize_chromosome)
        impu_test_mask = (impu_chromosomes == test_chromosome).to_numpy()

        if impu_test_mask.sum() == 0:
            raise ValueError(
                "%s contains no imputation rows from held-out chromosome %s."
                % (filename, test_chromosome)
            )

        dataset_im_test = dataset_impu.loc[impu_test_mask].copy()
        imputation_features = dataset_im_test.values[:, 2:]
        prediction = model.predict_proba(imputation_features)[:, 1]

        pre = np.full(len(prediction), np.nan)
        pre[prediction >= 0.6] = 1
        pre[prediction <= 0.4] = 0

        data_pre = pd.DataFrame(
            {
                "chrom": dataset_im_test.iloc[:, 0].values,
                "location": dataset_im_test.iloc[:, 1].values,
                "pre_meth": pre,
            }
        )

        output_prediction_path = os.path.join(
            file_dir_pre, "%s.txt" % filename.split(".")[0]
        )
        data_pre.to_csv(
            output_prediction_path,
            sep="\t",
            header=True,
            index=False,
        )
        print(
            "%s: %s imputation completed (%d rows)."
            % (
                filename.split(".")[0],
                test_chromosome,
                len(data_pre),
            )
        )

    result_table = pd.DataFrame(result_rows)
    safe_test_chromosome = re.sub(r"[^A-Za-z0-9_-]+", "_", test_chromosome)
    result_path = os.path.join(
        gse,
        "chromosome_holdout_catboost_%s.csv" % safe_test_chromosome,
    )
    result_table.to_csv(result_path, sep=",", header=True, index=False)
    print("Chromosome-based evaluation results saved to: %s" % result_path)
