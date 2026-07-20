import glob
import gzip
import math
import numpy as np
import os
from pathlib import Path
import datetime

# ======================== Global Configuration ========================
METHYLATION_THRESHOLD = 0.5
EXCLUDE_CHROM = "MT"         # Exclude MT chromosome
OUTPUT_DIR = "newdata_stat"
GLOBAL_PROGRESS_FILE = os.path.join(OUTPUT_DIR, "global_processing_progress.txt")
MAX_FILES_PER_DATASET = 300

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ======================== Dataset Configuration (Core) ========================
DATASETS_CONFIG = [
    {
        "name": "Li2021_GSE100272",
        "path_pattern": "GSE100272/*.bed.gz",
        "chrom_col": 0,
        "locus_col": 2,
        "meth_col": 5,
        "total_col": None,
        "file_format": "bed.gz",
        "data_location": "data/GSE100272/"
    }
]

# ---------------------- Global Progress Logging Function ----------------------
def log_global_progress(content):
    """Append to global progress log"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(GLOBAL_PROGRESS_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {content}\n")
    os.fsync(os.open(GLOBAL_PROGRESS_FILE, os.O_RDWR))
    print(f"Progress Log: {content}")

# ---------------------- Initialize File-level Data File (Write Header) ----------------------
def init_cell_detail_file(dataset_name):
    """Initialize file-level detail data file (write header only)"""
    cell_detail_file = os.path.join(OUTPUT_DIR, f"{dataset_name}_cell_details.csv")
    if not os.path.exists(cell_detail_file):
        with open(cell_detail_file, "w", encoding="utf-8") as f:
            f.write("file_name,total_observed_cpg,total_methylated_cpg,cell_methylation_rate\n")
        log_global_progress(f"{dataset_name} - Initialized file-level data file: {cell_detail_file}")
    return cell_detail_file

# ---------------------- Write Single File Data in Real-time ----------------------
def write_single_cell_data(dataset_name, file_name, file_stats):
    """
    Append single file data to file-level data file after processing
    :param dataset_name: Dataset name
    :param file_name: File name
    :param file_stats: File stats dict (total_observed_cpg, total_methylated_cpg, cell_methylation_rate)
    """
    cell_detail_file = init_cell_detail_file(dataset_name)
    with open(cell_detail_file, "a", encoding="utf-8") as f:
        f.write(
            f"{file_name},"
            f"{file_stats['total_observed_cpg']},"
            f"{file_stats['total_methylated_cpg']},"
            f"{file_stats['cell_methylation_rate']:.6f}\n"
        )
    os.fsync(os.open(cell_detail_file, os.O_RDWR))
    log_global_progress(f"{dataset_name} - Written file data: {file_name} | Methylation rate: {file_stats['cell_methylation_rate']:.4f}")

# ---------------------- Save Dataset Summary Statistics ----------------------
def save_dataset_summary(ds_config, ds_stats):
    """Save dataset summary statistics"""
    ds_name = ds_config["name"]
    summary_file = os.path.join(OUTPUT_DIR, f"{ds_name}_summary_stats.csv")
    
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(f"# {ds_name} Dataset Summary (First {MAX_FILES_PER_DATASET} files) | Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("metric,value\n")
        # Basic info
        f.write(f"dataset_name,{ds_name}\n")
        f.write(f"max_files_limit,{MAX_FILES_PER_DATASET}\n")
        # Core metrics only
        f.write(f"total_files_processed,{ds_stats['total_files']}\n")
        f.write(f"total_observed_cpg,{ds_stats['total_observed_cpg']}\n")
        f.write(f"total_methylated_cpg,{ds_stats['total_methylated_cpg']}\n")
        f.write(f"avg_cell_methylation_rate,{ds_stats['avg_cell_methylation_rate']:.6f}\n")
    
    log_global_progress(f"{ds_name} - Summary statistics saved: {summary_file}")
    print(f"{ds_name} - Summary statistics saved: {summary_file}")

# ---------------------- Process Single Dataset ----------------------
def process_dataset(ds_config):
    """
    Process single dataset (only first 300 files, write data immediately after each file)
    :param ds_config: Dataset config dict
    :return: Dataset statistics result
    """
    ds_name = ds_config["name"]
    log_global_progress(f"Start processing dataset: {ds_name} (First {MAX_FILES_PER_DATASET} files only)")
    
    try:
        # Parse config
        path_pattern = ds_config["path_pattern"]
        chrom_col = ds_config["chrom_col"]
        locus_col = ds_config["locus_col"]
        meth_col = ds_config["meth_col"]
        total_col = ds_config["total_col"]
        
        # Find matched files
        matched_files = glob.glob(path_pattern)
        if not matched_files:
            log_global_progress(f"{ds_name} - No matched files found (pattern: {path_pattern}), skipped")
            print(f"{ds_name} - No matched files found (pattern: {path_pattern}), skipped")
            return None
        
        # Only process first 300 files
        total_matched = len(matched_files)
        processed_files = matched_files[:MAX_FILES_PER_DATASET]
        actual_process_count = len(processed_files)
        
        log_global_progress(f"{ds_name} - Found {total_matched} files, will process first {actual_process_count} files (max {MAX_FILES_PER_DATASET})")
        print(f"\n{'='*60}")
        print(f"Start processing dataset: {ds_name} (First {MAX_FILES_PER_DATASET} files only)")
        print(f"Found {total_matched} files, processing first {actual_process_count} files")
        print(f"{'='*60}")

        # Initialize stats
        ds_stats = {
            "total_files": 0,
            "total_observed_cpg": 0,
            "total_methylated_cpg": 0,
            "avg_cell_methylation_rate": 0
        }
        file_methylation_rates = []

        # Process first 300 files
        for file_counter, file_path in enumerate(processed_files, 1):
            file_name = os.path.basename(file_path)
            log_global_progress(f"{ds_name} - Start processing file [{file_counter}/{actual_process_count}]：{file_name}")
            print(f"\n  [{file_counter}/{actual_process_count}] Processing file: {file_name}")

            # Current file stats
            total_observed_cpg = 0
            total_methylated_cpg = 0

            # Read compressed file
            with gzip.open(file_path, "rt", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    parts = line.split()
                    # Check column count
                    required_cols = [chrom_col, locus_col, meth_col]
                    if total_col is not None:
                        required_cols.append(total_col)
                    max_col = max(required_cols)
                    if len(parts) <= max_col:
                        continue

                    # Process chromosome
                    current_chrom = parts[chrom_col].strip()
                    if not current_chrom.startswith("chr"):
                        current_chrom = f"chr{current_chrom}"
                    if current_chrom == "chrMT" or current_chrom == "MT":
                        continue

                    # Extract locus (skip format check for speed)
                    try:
                        int(parts[locus_col])
                    except ValueError:
                        continue

                    # Extract methylation data
                    try:
                        if total_col is not None:
                            meth_value = float(parts[meth_col])
                            total_value = float(parts[total_col])
                            if total_value <= 0:
                                continue
                            meth_rate = meth_value / total_value
                        else:
                            meth_rate = float(parts[meth_col])
                    except ValueError:
                        continue

                    # Update core stats
                    total_observed_cpg += 1
                    if meth_rate > METHYLATION_THRESHOLD:
                        total_methylated_cpg += 1

            # Calculate methylation rate for current file
            cell_methylation_rate = (total_methylated_cpg / total_observed_cpg) if total_observed_cpg > 0 else 0
            
            # File stats
            file_stats = {
                "total_observed_cpg": total_observed_cpg,
                "total_methylated_cpg": total_methylated_cpg,
                "cell_methylation_rate": cell_methylation_rate
            }

            # Write single file data immediately
            write_single_cell_data(ds_name, file_name, file_stats)

            # Output core stats only
            print(f"     File statistics:")
            print(f"        • Total observed CPG: {total_observed_cpg}")
            print(f"        • Total methylated CPG: {total_methylated_cpg}")
            print(f"        • Cell methylation rate: {cell_methylation_rate:.4f}")

            # Update dataset stats
            ds_stats["total_files"] += 1
            ds_stats["total_observed_cpg"] += total_observed_cpg
            ds_stats["total_methylated_cpg"] += total_methylated_cpg
            file_methylation_rates.append(cell_methylation_rate)

        # Calculate average methylation rate
        if file_methylation_rates:
            ds_stats["avg_cell_methylation_rate"] = np.mean(file_methylation_rates)

        # Save dataset summary
        save_dataset_summary(ds_config, ds_stats)

        # Output core dataset stats
        print(f"\n{ds_name} Dataset Core Statistics (First {actual_process_count} files):")
        print(f"  - Processed files: {ds_stats['total_files']} (max {MAX_FILES_PER_DATASET})")
        print(f"  - Total observed CPG: {ds_stats['total_observed_cpg']}")
        print(f"  - Total methylated CPG: {ds_stats['total_methylated_cpg']}")
        print(f"  - Average cell methylation rate: {ds_stats['avg_cell_methylation_rate']:.4f}")

        log_global_progress(f"{ds_name} - Dataset processing completed (processed {ds_stats['total_files']} files)")
        return ds_stats

    except Exception as e:
        error_msg = f"{ds_name} - Processing error: {str(e)}"
        log_global_progress(error_msg)
        print(f"{error_msg}")
        return None

# ---------------------- Summarize All Datasets ----------------------
def save_all_datasets_summary(all_results):
    """Summarize all datasets results"""
    summary_file = os.path.join(OUTPUT_DIR, "ALL_DATASETS_FINAL_SUMMARY.csv")
    with open(summary_file, "w", encoding="utf-8") as f:
        headers = [
            "dataset_name", "max_files_limit",
            "total_files_processed", "total_observed_cpg",
            "total_methylated_cpg", "avg_cell_methylation_rate"
        ]
        f.write(",".join(headers) + "\n")
        
        for ds_config, ds_result in zip(DATASETS_CONFIG, all_results):
            if ds_result is None:
                continue
            row = [
                ds_config["name"],
                str(MAX_FILES_PER_DATASET),
                str(ds_result["total_files"]),
                str(ds_result["total_observed_cpg"]),
                str(ds_result["total_methylated_cpg"]),
                f"{ds_result['avg_cell_methylation_rate']:.6f}"
            ]
            f.write(",".join(row) + "\n")
    
    log_global_progress(f"All datasets summarized (First {MAX_FILES_PER_DATASET} files): {summary_file}")
    print(f"\nAll datasets summarized! Summary file: {summary_file}")
    print(f"All output files are saved in: {os.path.abspath(OUTPUT_DIR)}")

# ---------------------- Main Process ----------------------
if __name__ == "__main__":
    # Initialize global progress file
    with open(GLOBAL_PROGRESS_FILE, "w", encoding="utf-8") as f:
        f.write(f"{'='*50}\n")
        f.write(f"Batch processing started (First {MAX_FILES_PER_DATASET} files per dataset) - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*50}\n")
    
    print("="*70)
    print(f"Start batch processing methylation datasets (First {MAX_FILES_PER_DATASET} files per dataset)")
    print(f"Unified output directory: {os.path.abspath(OUTPUT_DIR)}")
    print(f"Global progress log: {GLOBAL_PROGRESS_FILE}")
    print("Core feature: Write data immediately after each file processed to prevent data loss")
    print("="*70)

    # Store all datasets results
    all_ds_results = []

    # Process each dataset
    for ds_config in DATASETS_CONFIG:
        ds_result = process_dataset(ds_config)
        all_ds_results.append(ds_result)

    # Summarize all results
    valid_results = [r for r in all_ds_results if r is not None]
    if valid_results:
        save_all_datasets_summary(all_ds_results)
    else:
        log_global_progress("No datasets processed successfully")
        print("\nNo datasets processed successfully!")

    # Log completion
    log_global_progress(f"All datasets processing completed (First {MAX_FILES_PER_DATASET} files per dataset)")
    print(f"\n" + "="*70)
    print(f"All datasets processing completed! (First {MAX_FILES_PER_DATASET} files per dataset)")
    print(f"Full progress log: {GLOBAL_PROGRESS_FILE}")
    print("="*70)