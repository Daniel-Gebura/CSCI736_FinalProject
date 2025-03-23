################################################################
# download_asl100_clips.py
#
# Description: Extracts ASL100 sign clips from YouTube videos
# using MASL-style JSON metadata files, clips them using ffmpeg,
# and stores them in a training-ready directory format.
#
# Author: Daniel Gebura
################################################################

import os
import json
import subprocess
from tqdm import tqdm  # For displaying progress bars

# Constants
JSON_DIR = "../msasl/MS-ASL"                  # Folder containing all MASL_*.json metadata files
OUTPUT_DIR = "../msasl/asl100_clips"          # Directory to save final extracted video clips
VIDEO_CACHE = "../msasl/video_cache"          # Directory to cache full downloaded YouTube videos
LABELS_FILE = "../data/asl100_labels.csv"     # Output CSV file mapping each clip to its label

# Ensure output directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VIDEO_CACHE, exist_ok=True)

def clean_url(url):
    """
    Ensure YouTube URL is well-formed and remove any unnecessary query parameters.

    Parameters:
        url (str): The raw URL string from the JSON metadata

    Returns:
        str: Cleaned URL string, prefixed with 'https://' and without query parameters
    """
    if not url.startswith("http"):
        url = "https://" + url
    return url.split("&")[0]  # Remove anything after '&' like &t=1s

def download_video(url, output_path):
    """
    Download the full YouTube video using yt-dlp if it's not already cached.

    Parameters:
        url (str): The cleaned YouTube URL
        output_path (str): The local path to save the downloaded video
    """
    if os.path.exists(output_path):
        return  # Skip if already downloaded

    # yt-dlp command to download the video in MP4 format
    command = [
        "yt-dlp",
        "-f", "mp4",
        "-o", output_path,
        url
    ]
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"yt-dlp failed for {url}: {e}")

def extract_clip(video_path, start, end, output_path):
    """
    Extract a subclip from the full video using ffmpeg.

    Parameters:
        video_path (str): Path to the full downloaded video
        start (float): Start time in seconds
        end (float): End time in seconds
        output_path (str): Path to save the extracted clip
    """
    command = [
        "ffmpeg",
        "-loglevel", "error",       # Suppress verbose output
        "-ss", str(start),          # Start time
        "-to", str(end),            # End time
        "-i", video_path,           # Input video
        "-c", "copy",               # Copy codecs without re-encoding
        "-y", output_path           # Overwrite output if exists
    ]
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError:
        print(f"Failed to extract clip to {output_path}")

def parse_float(value):
    """
    Safely convert a string or numeric value to float.

    Parameters:
        value (str or float): Value to convert

    Returns:
        float: Converted float value or 0.0 on failure
    """
    try:
        return float(value)
    except:
        return 0.0

def process_json_file(json_path, labels_output):
    """
    Processes one MASL JSON file and extracts all valid ASL100 clips from it.

    Parameters:
        json_path (str): Path to the .json metadata file
        labels_output (list): List to append (filename, label_text, label_id) tuples to
    """
    with open(json_path, "r") as f:
        entries = json.load(f)  # Load all metadata entries

    # Iterate over all entries in the JSON file
    for entry in tqdm(entries, desc=f"Processing {os.path.basename(json_path)}"):
        try:
            label_id = int(entry["label"])
            if label_id >= 107:
                continue  # Skip entries outside ASL100 label set

            # Extract relevant metadata fields
            label_text = entry["clean_text"].strip().lower().replace(" ", "_")  # Normalize label
            signer_id = entry["signer_id"]
            start_time = parse_float(entry["start_time"])
            end_time = parse_float(entry["end_time"])
            url = clean_url(entry["url"])

            # Create YouTube ID and cache path
            video_id = url.split("v=")[-1]
            full_video_path = os.path.join(VIDEO_CACHE, f"{video_id}.mp4")

            # Create a unique and descriptive output filename for the clip
            clip_filename = f"{signer_id}_{label_id}_{label_text}_{int(start_time*100):06d}-{int(end_time*100):06d}.mp4"
            clip_path = os.path.join(OUTPUT_DIR, clip_filename)

            if os.path.exists(clip_path):
                continue  # Skip if clip already extracted

            # Download the video and extract the clip
            download_video(url, full_video_path)
            if os.path.exists(full_video_path):
                extract_clip(full_video_path, start_time, end_time, clip_path)
                labels_output.append((clip_filename, label_text, label_id))  # Save label info

        except Exception as e:
            print(f"Error processing entry: {e}")  # Handle any metadata issues gracefully

def main():
    """
    Main pipeline to:
    - Iterate over all JSON metadata files
    - Extract all valid ASL100 video clips
    - Save clip-to-label mapping in a CSV file
    """
    # Collect all JSON files in the metadata directory
    json_files = [f for f in os.listdir(JSON_DIR) if f.endswith(".json")]
    all_labels = []  # List of (filename, label_text, label_id)

    # Process each JSON metadata file
    for json_file in json_files:
        full_path = os.path.join(JSON_DIR, json_file)
        process_json_file(full_path, all_labels)

    # Save all collected labels to a CSV file
    with open(LABELS_FILE, "w") as f:
        f.write("filename,label_text,label_id\n")
        for filename, label_text, label_id in all_labels:
            f.write(f"{filename},{label_text},{label_id}\n")

    print(f"\nSaved {len(all_labels)} ASL100 clips to: {OUTPUT_DIR}")
    print(f"Labels written to: {LABELS_FILE}")

# Entry point for the script
if __name__ == "__main__":
    main()