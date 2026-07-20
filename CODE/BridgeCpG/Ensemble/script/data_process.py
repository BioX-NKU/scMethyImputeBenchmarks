"""Convert the generated model NPZ files into aligned final CSV matrices."""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pandas as pd


CHR_NPZ = "/home/methyeval/data/lianght/cpg-transformer-main/y_ser.npz"
MAMBA_NPZ = "MambaCpG.npz"
TRANSFORMER_NPZ = "CpGTransformer.npz"
GRAPH_NPZ = "graphCpG.npz"

OUTPUT_DIR = Path("./processed_data")
M_DIFF_THRESHOLD = 100000
GRAPH_REMOVE_ROWS = 10
TRIM_LAST_ROWS = 1024


def get_split_row(chr_file: str) -> int:
    with np.load(chr_file, allow_pickle=True) as data:
        if "chr10" not in data:
            raise KeyError(f"chr10 not found in {chr_file}")
        chr10_rows = data["chr10"].shape[0]

    split_row = math.ceil(chr10_rows / 1024) * 1024
    print(f"chr10 row count: {chr10_rows}, split row count: {split_row}\n")
    return split_row


def save_matrix_csv(
    matrix: np.ndarray,
    output_prefix: str,
    value_name: str,
    part: int,
) -> Path | None:
    if matrix.size == 0:
        print(f"Warning: {value_name} Part {part} is empty; file not written.")
        return None

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{output_prefix}{value_name}_part{part}.csv"
    dataframe = pd.DataFrame(
        matrix,
        index=[f"row_{i}" for i in range(matrix.shape[0])],
        columns=[f"col_{i}" for i in range(matrix.shape[1])],
    )
    dataframe.to_csv(
        path,
        index=True,
        header=True,
        na_rep="NaN",
        encoding="utf-8",
    )
    print(f"{value_name} Part {part} saved to: {path}")
    return path


def process_single_dense_npz(
    file_path: str,
    pred_key: str,
    output_prefix: str,
    split_row: int,
) -> dict[str, tuple[int, int]]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"NPZ file does not exist: {file_path}")

    print(f"Processing file: {file_path}")
    with np.load(file_path, allow_pickle=True) as data:
        if "y_true" not in data:
            raise KeyError(f"y_true not found in {file_path}")
        if pred_key not in data:
            raise KeyError(f"{pred_key} not found in {file_path}")

        y_true = data["y_true"].astype(np.float32)
        y_pred = data[pred_key].astype(np.float32)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch in {file_path}: y_true={y_true.shape}, "
            f"{pred_key}={y_pred.shape}"
        )

    actual_split = min(split_row, y_true.shape[0])
    y_true_parts = (y_true[:actual_split], y_true[actual_split:])
    y_pred_parts = (y_pred[:actual_split], y_pred[actual_split:])

    print(
        f"Data shape: {y_true.shape}; Part 1 rows: {y_true_parts[0].shape[0]}; "
        f"Part 2 rows: {y_true_parts[1].shape[0]}"
    )

    for part, matrix in enumerate(y_true_parts, start=1):
        save_matrix_csv(matrix, output_prefix, "y_true", part)
    for part, matrix in enumerate(y_pred_parts, start=1):
        save_matrix_csv(matrix, output_prefix, pred_key, part)

    for part, (true_part, pred_part) in enumerate(
        zip(y_true_parts, y_pred_parts),
        start=1,
    ):
        print(
            f"Part {part}: y_true={true_part.shape}, "
            f"y_true NaNs={int(np.isnan(true_part).sum())}; "
            f"{pred_key}={pred_part.shape}, "
            f"prediction NaNs={int(np.isnan(pred_part).sum())}"
        )

    return {
        "part1_shape": y_true_parts[0].shape,
        "part2_shape": y_true_parts[1].shape,
    }


def split_graph_records(
    y: np.ndarray,
    m: np.ndarray,
    n: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    if not (len(y) == len(m) == len(n)):
        raise ValueError(
            "GraphCpG arrays have inconsistent lengths: "
            f"y={len(y)}, m={len(m)}, n={len(n)}"
        )

    m_diffs = np.diff(m)
    large_diff_idx = np.where(np.abs(m_diffs) > M_DIFF_THRESHOLD)[0]

    if len(large_diff_idx) > 0:
        split_idx = int(large_diff_idx[0] + 1)
        print(
            f"GraphCpG split at record {split_idx} "
            f"(absolute m difference={abs(m_diffs[large_diff_idx[0]])})"
        )
    else:
        split_idx = len(y) // 2
        print(
            "Warning: no large GraphCpG coordinate jump was found; "
            f"using midpoint {split_idx}."
        )

    return (
        {"y": y[:split_idx], "m": m[:split_idx], "n": n[:split_idx]},
        {"y": y[split_idx:], "m": m[split_idx:], "n": n[split_idx:]},
    )


def build_graph_matrix(data: dict[str, np.ndarray], output_path: Path) -> pd.DataFrame:
    if len(data["y"]) == 0:
        raise ValueError(f"No GraphCpG records available for {output_path}")

    all_m = np.asarray(data["m"])
    all_n = np.asarray(data["n"])
    all_y = np.asarray(data["y"])

    max_m = int(all_m.max())
    max_n = int(all_n.max()) if len(all_n) > 0 else 0
    full_matrix = np.full((max_m + 1, max_n + 1), np.nan, dtype=np.float32)

    valid_mask = all_m >= 11
    valid_m = all_m[valid_mask].astype(np.int64)
    valid_n = all_n[valid_mask].astype(np.int64)
    valid_y = all_y[valid_mask]

    for m_value, n_value, y_value in zip(valid_m, valid_n, valid_y):
        if 0 <= m_value <= max_m and 0 <= n_value <= max_n:
            full_matrix[m_value, n_value] = y_value

    if full_matrix.shape[0] > GRAPH_REMOVE_ROWS:
        matrix_after_remove = full_matrix[GRAPH_REMOVE_ROWS:]
    else:
        matrix_after_remove = full_matrix

    # Preserve the notebook mapping: m=11 becomes row_0.
    final_matrix = (
        matrix_after_remove[1:]
        if matrix_after_remove.shape[0] > 1
        else matrix_after_remove
    )

    dataframe = pd.DataFrame(
        final_matrix,
        index=[f"row_{i}" for i in range(final_matrix.shape[0])],
        columns=[f"col_{i}" for i in range(final_matrix.shape[1])],
    )
    dataframe.to_csv(
        output_path,
        index=True,
        header=True,
        na_rep="NaN",
        encoding="utf-8",
    )
    print(f"GraphCpG matrix saved to {output_path}: {final_matrix.shape}")
    return dataframe


def process_graph_npz(file_path: str) -> None:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"NPZ file does not exist: {file_path}")

    with np.load(file_path, allow_pickle=True) as data:
        required = {"y_pred_sigmoid", "m_global", "n_global"}
        missing = sorted(required.difference(data.files))
        if missing:
            raise KeyError(f"{file_path} is missing keys: {missing}")

        y = np.asarray(data["y_pred_sigmoid"])
        m = np.asarray(data["m_global"])
        n = np.asarray(data["n_global"])

    part1, part2 = split_graph_records(y, m, n)
    build_graph_matrix(part1, OUTPUT_DIR / "graphCpG_part1.csv")
    build_graph_matrix(part2, OUTPUT_DIR / "graphCpG_part2.csv")


def adjust_csv_rows(file_path: Path, target_rows: int) -> None:
    if target_rows <= 0:
        raise ValueError(f"Invalid target row count {target_rows} for {file_path}")

    dataframe = pd.read_csv(file_path, index_col=0)
    if len(dataframe) > target_rows:
        dataframe.iloc[:target_rows].to_csv(
            file_path,
            index=True,
            header=True,
            na_rep="NaN",
            encoding="utf-8",
        )
        print(f"Trimmed {file_path.name} to {target_rows} rows.")
    elif len(dataframe) < target_rows:
        raise ValueError(
            f"{file_path.name} has only {len(dataframe)} rows, "
            f"fewer than target {target_rows}."
        )


def trim_all_part_csvs(target_part1: int, target_part2: int) -> None:
    for file_path in OUTPUT_DIR.glob("*.csv"):
        lower_name = file_path.name.lower()
        if "part1" in lower_name:
            adjust_csv_rows(file_path, target_part1)
        elif "part2" in lower_name:
            adjust_csv_rows(file_path, target_part2)


def sync_nan_ignore_index(
    source_file: Path,
    target_file: Path,
    backup_suffix: str = "_backup",
) -> None:
    source = pd.read_csv(source_file)
    target = pd.read_csv(target_file)

    if source.shape[1] != target.shape[1]:
        raise ValueError(
            f"Column count mismatch: {source_file.name}={source.shape[1]}, "
            f"{target_file.name}={target.shape[1]}"
        )
    if len(source) != len(target):
        raise ValueError(
            f"Row count mismatch: {source_file.name}={len(source)}, "
            f"{target_file.name}={len(target)}"
        )

    source_data = source.iloc[:, 1:]
    target_data = target.iloc[:, 1:].copy()
    nan_mask = source_data.isna()

    if int(nan_mask.sum().sum()) == 0:
        print(f"No NaN synchronization required for {target_file.name}.")
        return

    backup_file = target_file.with_name(
        f"{target_file.stem}{backup_suffix}{target_file.suffix}"
    )
    target.to_csv(
        backup_file,
        index=False,
        header=True,
        na_rep="NaN",
        encoding="utf-8",
    )

    target_data = target_data.mask(nan_mask)
    target.iloc[:, 1:] = target_data
    target.to_csv(
        target_file,
        index=False,
        header=True,
        na_rep="NaN",
        encoding="utf-8",
    )
    print(
        f"Synchronized NaNs: {source_file.name} -> {target_file.name}; "
        f"backup: {backup_file.name}"
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    split_row = get_split_row(CHR_NPZ)

    mamba_shapes = process_single_dense_npz(
        MAMBA_NPZ,
        pred_key="y_pred_prob",
        output_prefix="mamba_",
        split_row=split_row,
    )
    process_single_dense_npz(
        TRANSFORMER_NPZ,
        pred_key="y_pred_prob",
        output_prefix="transformer_",
        split_row=split_row,
    )
    process_graph_npz(GRAPH_NPZ)

    target_part1 = mamba_shapes["part1_shape"][0] - TRIM_LAST_ROWS
    target_part2 = mamba_shapes["part2_shape"][0] - TRIM_LAST_ROWS
    trim_all_part_csvs(target_part1, target_part2)

    sync_nan_ignore_index(
        OUTPUT_DIR / "transformer_y_true_part1.csv",
        OUTPUT_DIR / "graphCpG_part1.csv",
    )
    sync_nan_ignore_index(
        OUTPUT_DIR / "transformer_y_true_part2.csv",
        OUTPUT_DIR / "graphCpG_part2.csv",
    )

    print("\nData processing completed.")


if __name__ == "__main__":
    main()
