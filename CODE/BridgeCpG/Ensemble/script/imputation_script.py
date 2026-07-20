"""Apply the trained ensemble classifier to Part 1 and save final scores."""

from __future__ import annotations

import os

import joblib
import pandas as pd

from feature_extractor import load_clean_data, prepare_samples
from classification_script import (
    calculate_metrics,
    create_comparison_table,
    plot_roc_comparison,
)


file_paths_test = {
    "graph": os.path.join("processed_data", "graphCpG_part1.csv"),
    "mamba": os.path.join("processed_data", "mamba_y_pred_prob_part1.csv"),
    "transformer": os.path.join(
        "processed_data", "transformer_y_pred_prob_part1.csv"
    ),
    "true": os.path.join("processed_data", "transformer_y_true_part1.csv"),
}

output_dir = "./meta_results"
model_path = os.path.join(
    output_dir,
    "ensemble_classifier_balanced_f1_acc.pkl",
)


def main() -> None:
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Trained model not found: {model_path}. "
                "Run classification_script.py first."
            )

        print("Applying trained ensemble classifier to Part 1.")

        print("\n1. Loading trained classifier")
        ensemble_clf_info = joblib.load(model_path)

        print("\n2. Loading Part 1 data")
        data_test = load_clean_data(file_paths_test, "part1 (Test)")

        print("\n3. Extracting neighborhood features")
        test_samples_df = prepare_samples(data_test, "part1 (Test)")

        print("\n4. Generating ensemble scores and metrics")
        metrics_test, test_samples_with_pred = calculate_metrics(
            test_samples_df,
            ensemble_clf_info,
            "part1 (Test)",
        )

        print("\nTest Set Metrics (part1):")
        print(metrics_test.to_string())

        test_result_path = os.path.join(
            output_dir,
            "part1_test_detailed_results_balanced.csv",
        )
        test_samples_with_pred.to_csv(test_result_path, index=False)
        metrics_test.to_csv(
            os.path.join(
                output_dir,
                "part1_test_metrics_balanced.csv",
            )
        )

        compact_columns = [
            column
            for column in [
                "col_idx",
                "pos_idx",
                "y_true",
                "graph_current",
                "mamba_current",
                "transformer_current",
                "ensemble_score",
                "ensemble_pred",
                "ensemble_best_thresh",
            ]
            if column in test_samples_with_pred.columns
        ]
        test_samples_with_pred[compact_columns].to_csv(
            os.path.join(
                output_dir,
                "part1_ensemble_imputation_results.csv",
            ),
            index=False,
        )

        plot_roc_comparison(
            test_samples_with_pred,
            output_dir,
            "part1 (Test)",
        )

        train_metrics_path = os.path.join(
            output_dir,
            "part2_training_metrics_balanced.csv",
        )
        if os.path.exists(train_metrics_path):
            metrics_train = pd.read_csv(train_metrics_path, index_col=0)
            comparison_df = create_comparison_table(
                metrics_train,
                metrics_test,
            )
            comparison_df.to_csv(
                os.path.join(
                    output_dir,
                    "cross_part_comparison_balanced.csv",
                )
            )
            print("\nCross-Part Comparison:")
            print(comparison_df.to_string())
        else:
            print(
                "\nTraining metrics were not found; "
                "cross-part comparison was skipped."
            )

        print("\nImputation/application step completed.")
        print(f"Part 1 samples: {len(test_samples_df)}")
        print(f"Results saved to: {os.path.abspath(output_dir)}")

    except Exception as error:
        print(f"\nProcess interrupted: {error}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
