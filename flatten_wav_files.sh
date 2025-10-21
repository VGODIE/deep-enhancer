#!/bin/bash

# Script to move all WAV files from nested directories to a top-level directory
# Usage: ./flatten_wav_files.sh <source_directory> [output_directory] [--clean]

set -e

# Parse arguments
CLEAN_SUBDIRS=false
SOURCE_DIR=""
OUTPUT_DIR=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --clean)
            CLEAN_SUBDIRS=true
            shift
            ;;
        *)
            if [ -z "$SOURCE_DIR" ]; then
                SOURCE_DIR="$1"
            elif [ -z "$OUTPUT_DIR" ]; then
                OUTPUT_DIR="$1"
            fi
            shift
            ;;
    esac
done

# Check if source directory is provided
if [ -z "$SOURCE_DIR" ]; then
    echo "Usage: $0 <source_directory> [output_directory] [--clean]"
    echo ""
    echo "Options:"
    echo "  --clean    Remove subdirectories from source after copying WAV files"
    echo ""
    echo "Examples:"
    echo "  $0 dns_clean ./wav_files"
    echo "  $0 dns_clean --clean"
    echo "  $0 dns_clean ./wav_files --clean"
    exit 1
fi

OUTPUT_DIR="${OUTPUT_DIR:-${SOURCE_DIR}_flattened}"

# Check if source directory exists
if [ ! -d "$SOURCE_DIR" ]; then
    echo "Error: Source directory '$SOURCE_DIR' does not exist"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

echo "Searching for WAV files in: $SOURCE_DIR"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Count total WAV files
total_files=$(find "$SOURCE_DIR" -type f -iname "*.wav" | wc -l | tr -d ' ')
echo "Found $total_files WAV files"
echo ""

# Counter for processed files
counter=0

# Find all WAV files and copy them to output directory
find "$SOURCE_DIR" -type f -iname "*.wav" | while read -r file; do
    counter=$((counter + 1))

    # Get just the filename
    filename=$(basename "$file")

    # Check if file already exists in output directory
    if [ -f "$OUTPUT_DIR/$filename" ]; then
        # If filename conflicts, add a unique identifier
        filename_no_ext="${filename%.wav}"
        # Use a hash of the full path to create unique name
        unique_id=$(echo "$file" | md5sum | cut -c1-8)
        new_filename="${filename_no_ext}_${unique_id}.wav"

        echo "[$counter/$total_files] Conflict detected, renaming: $filename -> $new_filename"
        cp "$file" "$OUTPUT_DIR/$new_filename"
    else
        echo "[$counter/$total_files] Copying: $filename"
        cp "$file" "$OUTPUT_DIR/$filename"
    fi
done

echo ""
echo "Done! All WAV files copied to: $OUTPUT_DIR"
echo "Total files in output directory: $(find "$OUTPUT_DIR" -type f -name "*.wav" | wc -l | tr -d ' ')"

# Clean up subdirectories if requested
if [ "$CLEAN_SUBDIRS" = true ]; then
    echo ""
    echo "Cleaning up subdirectories in: $SOURCE_DIR"

    # Find and remove all subdirectories (but keep WAV files in root of SOURCE_DIR if any)
    find "$SOURCE_DIR" -mindepth 1 -type d -exec rm -rf {} + 2>/dev/null || true

    # Count remaining items
    remaining_dirs=$(find "$SOURCE_DIR" -mindepth 1 -type d | wc -l | tr -d ' ')
    remaining_files=$(find "$SOURCE_DIR" -type f | wc -l | tr -d ' ')

    echo "Cleanup complete!"
    echo "Remaining subdirectories in $SOURCE_DIR: $remaining_dirs"
    echo "Remaining files in $SOURCE_DIR: $remaining_files"
fi
