"""
Turkey Weather Data Consolidation - ETL Merger
----------------------------------------------
Author: Miraç Çelikel
Description:
    This script serves as the final step in the Data Collection Pipeline.
    It performs the 'Aggregation' and 'Transformation' phases of the ETL process.

    Functionality:
    1. EXTRACT: Scans the 'city_weather_data' directory for individual city shards (CSVs).
    2. TRANSFORM:
        - Merges 81+ individual files into a single Pandas DataFrame.
        - Enforces datetime data types for temporal analysis.
        - Sorts data hierarchically by Plate Code (City ID) and Date.
    3. LOAD: Exports the cleaned, consolidated dataset to 'Turkey_Weather_Master.csv'.

Usage:
    Run this script after all batch scripts (batch_01, batch_02, batch_03) have completed.
"""

import pandas as pd
import os
import glob
import sys

# --- CONFIGURATION ---
# Paths are relative to the script's location
INPUT_FOLDER = "../data/city_weather_data"
OUTPUT_FILE = "../data/Turkey_Weather_Master.csv"


def main():
    print("Starting Data Consolidation (Merge) Process...")

    # --- STEP 1: SETUP & DISCOVERY ---
    # Determine absolute paths to ensure script runs correctly from any directory
    base_path = os.path.dirname(os.path.abspath(__file__))
    target_folder = os.path.join(base_path, INPUT_FOLDER)
    output_path = os.path.join(base_path, OUTPUT_FILE)

    # Find all CSV files in the target directory
    csv_files = glob.glob(os.path.join(target_folder, "*.csv"))

    if not csv_files:
        print(f"CRITICAL ERROR: No CSV files found in '{target_folder}'")
        print("-> Tip: Did you run the 'batch_XX.py' scripts first?")
        sys.exit(1)

    print(f"Found {len(csv_files)} city data files. Beginning aggregation...")

    # --- STEP 2: EXTRACTION (Reading Files) ---
    data_frames = []
    success_count = 0

    for file in csv_files:
        try:
            # Read individual shard
            df = pd.read_csv(file)
            data_frames.append(df)
            success_count += 1

            # Optional: Print progress for large datasets
            # print(f"   -> Loaded {os.path.basename(file)}", end="\r")

        except Exception as e:
            print(f"\nWARNING: Corrupt file detected: {os.path.basename(file)}")
            print(f"-> Error: {e}")

    print(f"\nSuccessfully loaded {success_count} / {len(csv_files)} files.")

    # --- STEP 3: TRANSFORMATION (Cleaning & Sorting) ---
    print("Processing and Cleaning data...")

    if not data_frames:
        print("Error: No valid data frames to merge.")
        return

    # Concatenate all shards into one Master DataFrame
    master_df = pd.concat(data_frames, ignore_index=True)

    # Type Casting: Ensure 'date' is actual datetime object, not string
    if 'date' in master_df.columns:
        master_df['date'] = pd.to_datetime(master_df['date'])

        # Hierarchical Sorting:
        # Primary Key: Plate Code (City), Secondary Key: Date (Chronological)
        master_df = master_df.sort_values(by=['plate_code', 'date'])

    print(f"   -> Merged Dataset Shape: {master_df.shape} (Rows, Columns)")

    # --- STEP 4: LOADING (Export) ---
    try:
        master_df.to_csv(output_path, index=False)
        print(f"SUCCESS! Master dataset generated successfully.")
        print(f"-> Location: {output_path}")
    except PermissionError:
        print(f"Error: Permission denied. Is '{OUTPUT_FILE}' open in another program?")


if __name__ == "__main__":
    main()