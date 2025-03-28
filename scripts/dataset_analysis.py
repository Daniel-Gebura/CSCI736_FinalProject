################################################################
# dataset_analysis.py
#
# Description: Analyze a CSV label file for dataset statistics,
# including class distribution, duplicates, and label frequency.
# Writes output to a formatted .txt file.
#
# Author: Daniel Gebura
################################################################

import pandas as pd

# Input and output file paths
LABEL_CSV_PATH = "../data/asl100_labels.csv"
OUTPUT_STATS_PATH = "../data/dataset_stats.txt"

def main():
    # Load the CSV label data
    df = pd.read_csv(LABEL_CSV_PATH)

    # Count total clips and unique labels
    total_clips = len(df)
    unique_labels = df["label_text"].unique()
    num_classes = len(unique_labels)

    # Count label occurrences
    label_counts = df["label_text"].value_counts()
    most_common = label_counts.head(5)
    least_common = label_counts.tail(5)

    # Check for duplicate filenames
    duplicate_filenames = df[df.duplicated("filename")]

    # Start writing the report
    with open(OUTPUT_STATS_PATH, "w") as f:
        f.write("DATASET STATISTICS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total clips: {total_clips}\n")
        f.write(f"Unique classes: {num_classes}\n\n")

        f.write("Top 5 most frequent labels:\n")
        f.write(most_common.to_string() + "\n\n")

        f.write("Bottom 5 least frequent labels:\n")
        f.write(least_common.to_string() + "\n\n")

        f.write("Label distribution (full):\n")
        f.write(label_counts.sort_values(ascending=False).to_string() + "\n\n")

        if not duplicate_filenames.empty:
            f.write("Duplicate filename entries found:\n")
            f.write(duplicate_filenames.to_string(index=False) + "\n")
        else:
            f.write("No duplicate filename entries found.\n")

    print(f"Dataset statistics written to: {OUTPUT_STATS_PATH}")

if __name__ == "__main__":
    main()
