"""Audit ordered alignment of the final CSV matrices."""

from __future__ import annotations

import gc
import hashlib
import os

import numpy as np
import pandas as pd


DATA_DIR = "./processed_data"
OUTPUT_DIR = "./meta_results"

CSV_FILES = {
    "part1": {
        "graph": "graphCpG_part1.csv",
        "mamba_pred": "mamba_y_pred_prob_part1.csv",
        "transformer_pred": "transformer_y_pred_prob_part1.csv",
        "mamba_true": "mamba_y_true_part1.csv",
        "transformer_true": "transformer_y_true_part1.csv",
    },
    "part2": {
        "graph": "graphCpG_part2.csv",
        "mamba_pred": "mamba_y_pred_prob_part2.csv",
        "transformer_pred": "transformer_y_pred_prob_part2.csv",
        "mamba_true": "mamba_y_true_part2.csv",
        "transformer_true": "transformer_y_true_part2.csv",
    },
}

SKIP_FIRST_DATA_ROW = True
SKIP_FIRST_DATA_COLUMN = True


def read_final_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Final CSV not found: {path}")

    dataframe = pd.read_csv(path, header=0, index_col=0)
    dataframe.index = dataframe.index.map(str)
    dataframe.columns = dataframe.columns.map(str)

    if dataframe.index.has_duplicates:
        duplicated = dataframe.index[
            dataframe.index.duplicated()
        ].tolist()[:5]
        raise AssertionError(
            f"{os.path.basename(path)} contains duplicated row labels: "
            f"{duplicated}"
        )
    if dataframe.columns.has_duplicates:
        duplicated = dataframe.columns[
            dataframe.columns.duplicated()
        ].tolist()[:5]
        raise AssertionError(
            f"{os.path.basename(path)} contains duplicated column labels: "
            f"{duplicated}"
        )

    return dataframe


def first_sequence_difference(left: list[str], right: list[str]):
    limit = min(len(left), len(right))
    for position in range(limit):
        if left[position] != right[position]:
            return position, left[position], right[position]

    if len(left) != len(right):
        left_value = left[limit] if limit < len(left) else "<END>"
        right_value = right[limit] if limit < len(right) else "<END>"
        return limit, left_value, right_value

    return None


def assert_same_ordered_layout(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    reference_name: str,
    candidate_name: str,
) -> None:
    if reference.shape != candidate.shape:
        raise AssertionError(
            f"{candidate_name} and {reference_name} have different shapes: "
            f"{candidate.shape} vs {reference.shape}"
        )

    row_difference = first_sequence_difference(
        reference.index.tolist(),
        candidate.index.tolist(),
    )
    if row_difference is not None:
        position, reference_value, candidate_value = row_difference
        raise AssertionError(
            f"Row-order mismatch between {reference_name} and "
            f"{candidate_name} at row position {position}: "
            f"{reference_value!r} vs {candidate_value!r}"
        )

    column_difference = first_sequence_difference(
        reference.columns.tolist(),
        candidate.columns.tolist(),
    )
    if column_difference is not None:
        position, reference_value, candidate_value = column_difference
        raise AssertionError(
            f"Column-order mismatch between {reference_name} and "
            f"{candidate_name} at column position {position}: "
            f"{reference_value!r} vs {candidate_value!r}"
        )


def first_value_difference(
    left: pd.DataFrame,
    right: pd.DataFrame,
):
    left_array = left.to_numpy(dtype=np.float64)
    right_array = right.to_numpy(dtype=np.float64)

    equal_mask = np.isclose(
        left_array,
        right_array,
        rtol=0.0,
        atol=0.0,
        equal_nan=True,
    )
    if np.all(equal_mask):
        return None

    row_position, column_position = np.argwhere(~equal_mask)[0]
    return {
        "row_position": int(row_position),
        "column_position": int(column_position),
        "row_label": left.index[row_position],
        "column_label": left.columns[column_position],
        "left_value": left_array[row_position, column_position],
        "right_value": right_array[row_position, column_position],
    }


def ordered_ground_truth_digest(dataframe: pd.DataFrame) -> str:
    values = dataframe.to_numpy(dtype=np.float64)
    nan_mask = np.isnan(values)
    normalized = np.nan_to_num(
        values,
        nan=9.87654321e307,
        posinf=8.76543210e307,
        neginf=-8.76543210e307,
    )

    digest = hashlib.sha256()
    digest.update(str(dataframe.shape).encode("utf-8"))
    digest.update("\n".join(dataframe.index.tolist()).encode("utf-8"))
    digest.update("\n".join(dataframe.columns.tolist()).encode("utf-8"))
    digest.update(nan_mask.tobytes())
    digest.update(np.ascontiguousarray(normalized).tobytes())
    return digest.hexdigest()


def get_actually_used_region(dataframe: pd.DataFrame) -> pd.DataFrame:
    row_start = 1 if SKIP_FIRST_DATA_ROW else 0
    column_start = 1 if SKIP_FIRST_DATA_COLUMN else 0
    return dataframe.iloc[row_start:, column_start:]


def main() -> pd.DataFrame:
    print("=" * 78)
    print("FINAL CSV ORDERED-ALIGNMENT AUDIT")
    print("Only ./processed_data/*.csv files are being checked.")
    print("=" * 78)

    audit_rows = []

    for part_name, filenames in CSV_FILES.items():
        print(f"\n[{part_name.upper()}] Loading final CSV files")

        dataframes = {
            name: read_final_csv(os.path.join(DATA_DIR, filename))
            for name, filename in filenames.items()
        }

        reference_name = "transformer_true"
        reference = dataframes[reference_name]

        for name, dataframe in dataframes.items():
            assert_same_ordered_layout(
                reference,
                dataframe,
                reference_name,
                name,
            )

        print(
            "[PASS] All five raw CSVs have exactly the same shape, "
            "row-label order and column-label order: "
            f"{reference.shape}"
        )

        true_difference = first_value_difference(
            dataframes["transformer_true"],
            dataframes["mamba_true"],
        )
        if true_difference is not None:
            raise AssertionError(
                "MambaCpG and CpGTransformer y_true CSVs are not "
                "identical. First mismatch: row position "
                f"{true_difference['row_position']} "
                f"({true_difference['row_label']}), column position "
                f"{true_difference['column_position']} "
                f"({true_difference['column_label']}), "
                f"Transformer={true_difference['left_value']}, "
                f"Mamba={true_difference['right_value']}."
            )

        print(
            "[PASS] MambaCpG and CpGTransformer y_true CSVs are "
            "identical in shape, NaN positions, values and element order."
        )

        used = {
            name: get_actually_used_region(dataframe)
            for name, dataframe in dataframes.items()
        }
        used_reference = used[reference_name]

        for name, dataframe in used.items():
            assert_same_ordered_layout(
                used_reference,
                dataframe,
                f"{reference_name} used region",
                f"{name} used region",
            )

        used_true_difference = first_value_difference(
            used["transformer_true"],
            used["mamba_true"],
        )
        if used_true_difference is not None:
            raise AssertionError(
                "The y_true matrices differ in the region actually used "
                "for training/evaluation."
            )

        print(
            "[PASS] After applying the same iloc slicing as the "
            "ensemble code, all model matrices remain identically "
            f"ordered: {used_reference.shape}"
        )

        true_values = used["transformer_true"].to_numpy(dtype=np.float64)
        graph_values = used["graph"].to_numpy(dtype=np.float64)
        mamba_values = used["mamba_pred"].to_numpy(dtype=np.float64)
        transformer_values = used["transformer_pred"].to_numpy(
            dtype=np.float64
        )

        observed_truth = ~np.isnan(true_values)
        graph_available = ~np.isnan(graph_values)
        mamba_available = ~np.isnan(mamba_values)
        transformer_available = ~np.isnan(transformer_values)

        joint_valid = (
            observed_truth
            & graph_available
            & mamba_available
            & transformer_available
        )

        non_binary = true_values[
            observed_truth & ~np.isin(true_values, [0.0, 1.0])
        ]
        if non_binary.size:
            raise AssertionError(
                f"{part_name}: y_true contains non-binary observed "
                f"values, for example {np.unique(non_binary)[:10]}"
            )

        digest = ordered_ground_truth_digest(
            used["transformer_true"]
        )

        print("[PASS] Observed y_true values are binary.")
        print(f"       Ordered y_true SHA-256: {digest}")
        print(
            f"       Observed y_true positions: "
            f"{int(observed_truth.sum()):,}"
        )
        print(
            "       GraphCpG available at observed positions: "
            f"{int((observed_truth & graph_available).sum()):,}"
        )
        print(
            "       MambaCpG available at observed positions: "
            f"{int((observed_truth & mamba_available).sum()):,}"
        )
        print(
            "       CpGTransformer available at observed positions: "
            f"{int((observed_truth & transformer_available).sum()):,}"
        )
        print(
            "       Position-wise common samples actually usable by all "
            f"models: {int(joint_valid.sum()):,}"
        )

        audit_rows.append({
            "part": part_name,
            "raw_rows": reference.shape[0],
            "raw_columns": reference.shape[1],
            "used_rows": used_reference.shape[0],
            "used_columns": used_reference.shape[1],
            "observed_y_true": int(observed_truth.sum()),
            "joint_valid_samples": int(joint_valid.sum()),
            "ordered_y_true_sha256": digest,
            "status": "PASS",
        })

        del dataframes, used
        gc.collect()

    audit_dataframe = pd.DataFrame(audit_rows)

    print("\n" + "=" * 78)
    print("FINAL RESULT: PASS")
    print(
        "The final CSVs use the same ordered row-column grid, and "
        "the independently exported MambaCpG/CpGTransformer "
        "ground-truth matrices are exactly identical."
    )
    print(
        "No sorting, merging or implicit reindexing was used in this audit."
    )
    print("=" * 78)
    print(audit_dataframe.to_string(index=False))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    audit_path = os.path.join(
        OUTPUT_DIR,
        "final_csv_alignment_audit.csv",
    )
    audit_dataframe.to_csv(audit_path, index=False)
    print(f"\nAudit table saved to: {audit_path}")

    return audit_dataframe


if __name__ == "__main__":
    main()
