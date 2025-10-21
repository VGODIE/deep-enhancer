#!/bin/bash

# Download DNS Challenge datasets using azcopy
# This script uses azcopy for faster, more reliable downloads from Azure Blob Storage

set -e  # Exit on error

# Dataset structure:
# datasets_fullband
# \-- clean_fullband 827G
#     +-- emotional_speech 2.4G
#     +-- french_speech 62G
#     +-- german_speech 319G
#     +-- italian_speech 42G
#     +-- read_speech 299G
#     +-- russian_speech 12G
#     +-- spanish_speech 65G
#     +-- vctk_wav48_silence_trimmed 27G
#     \-- VocalSet_48kHz_mono 974M

BLOB_NAMES=(
    clean_fullband/datasets_fullband.clean_fullband.VocalSet_48kHz_mono_000_NA_NA.tar.bz2
    #clean_fullband/datasets_fullband.clean_fullband.emotional_speech_000_NA_NA.tar.bz2

    clean_fullband/datasets_fullband.clean_fullband.french_speech_000_NA_NA.tar.bz2
    clean_fullband/datasets_fullband.clean_fullband.french_speech_001_NA_NA.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.french_speech_002_NA_NA.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.french_speech_003_NA_NA.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.french_speech_004_NA_NA.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.french_speech_005_NA_NA.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.french_speech_006_NA_NA.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.french_speech_007_NA_NA.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.french_speech_008_NA_NA.tar.bz2

    clean_fullband/datasets_fullband.clean_fullband.german_speech_000_0.00_3.47.tar.bz2
    clean_fullband/datasets_fullband.clean_fullband.german_speech_001_3.47_3.64.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.german_speech_002_3.64_3.74.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.german_speech_003_3.74_3.81.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.german_speech_004_3.81_3.86.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.german_speech_005_3.86_3.91.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.german_speech_006_3.91_3.96.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.german_speech_007_3.96_4.00.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.german_speech_008_4.00_4.04.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.german_speech_009_4.04_4.08.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.german_speech_010_4.08_4.12.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.german_speech_011_4.12_4.16.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.german_speech_012_4.16_4.21.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.german_speech_013_4.21_4.26.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.german_speech_014_4.26_4.33.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.german_speech_015_4.33_4.43.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.german_speech_016_4.43_NA.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.german_speech_017_NA_NA.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.german_speech_018_NA_NA.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.german_speech_019_NA_NA.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.german_speech_020_NA_NA.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.german_speech_021_NA_NA.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.german_speech_022_NA_NA.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.german_speech_023_NA_NA.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.german_speech_024_NA_NA.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.german_speech_025_NA_NA.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.german_speech_026_NA_NA.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.german_speech_027_NA_NA.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.german_speech_028_NA_NA.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.german_speech_029_NA_NA.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.german_speech_030_NA_NA.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.german_speech_031_NA_NA.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.german_speech_032_NA_NA.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.german_speech_033_NA_NA.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.german_speech_034_NA_NA.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.german_speech_035_NA_NA.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.german_speech_036_NA_NA.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.german_speech_037_NA_NA.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.german_speech_038_NA_NA.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.german_speech_039_NA_NA.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.german_speech_040_NA_NA.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.german_speech_041_NA_NA.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.german_speech_042_NA_NA.tar.bz2

    clean_fullband/datasets_fullband.clean_fullband.italian_speech_000_0.00_3.98.tar.bz2
    clean_fullband/datasets_fullband.clean_fullband.italian_speech_001_3.98_4.21.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.italian_speech_002_4.21_4.40.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.italian_speech_003_4.40_NA.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.italian_speech_004_NA_NA.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.italian_speech_005_NA_NA.tar.bz2

    clean_fullband/datasets_fullband.clean_fullband.read_speech_000_0.00_3.75.tar.bz2
    clean_fullband/datasets_fullband.clean_fullband.read_speech_001_3.75_3.88.tar.bz2
    clean_fullband/datasets_fullband.clean_fullband.read_speech_002_3.88_3.96.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.read_speech_003_3.96_4.02.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.read_speech_004_4.02_4.06.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.read_speech_005_4.06_4.10.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.read_speech_006_4.10_4.13.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.read_speech_007_4.13_4.16.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.read_speech_008_4.16_4.19.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.read_speech_009_4.19_4.21.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.read_speech_010_4.21_4.24.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.read_speech_011_4.24_4.26.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.read_speech_012_4.26_4.29.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.read_speech_013_4.29_4.31.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.read_speech_014_4.31_4.33.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.read_speech_015_4.33_4.35.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.read_speech_016_4.35_4.38.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.read_speech_017_4.38_4.40.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.read_speech_018_4.40_4.42.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.read_speech_019_4.42_4.45.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.read_speech_020_4.45_4.48.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.read_speech_021_4.48_4.52.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.read_speech_022_4.52_4.57.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.read_speech_023_4.57_4.67.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.read_speech_024_4.67_NA.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.read_speech_025_NA_NA.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.read_speech_026_NA_NA.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.read_speech_027_NA_NA.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.read_speech_028_NA_NA.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.read_speech_029_NA_NA.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.read_speech_030_NA_NA.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.read_speech_031_NA_NA.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.read_speech_032_NA_NA.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.read_speech_033_NA_NA.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.read_speech_034_NA_NA.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.read_speech_035_NA_NA.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.read_speech_036_NA_NA.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.read_speech_037_NA_NA.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.read_speech_038_NA_NA.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.read_speech_039_NA_NA.tar.bz2

    clean_fullband/datasets_fullband.clean_fullband.russian_speech_000_0.00_4.31.tar.bz2
    clean_fullband/datasets_fullband.clean_fullband.russian_speech_001_4.31_NA.tar.bz2

    clean_fullband/datasets_fullband.clean_fullband.spanish_speech_000_0.00_4.09.tar.bz2
    clean_fullband/datasets_fullband.clean_fullband.spanish_speech_001_4.09_NA.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.spanish_speech_002_NA_NA.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.spanish_speech_003_NA_NA.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.spanish_speech_004_NA_NA.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.spanish_speech_005_NA_NA.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.spanish_speech_006_NA_NA.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.spanish_speech_007_NA_NA.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.spanish_speech_008_NA_NA.tar.bz2

    clean_fullband/datasets_fullband.clean_fullband.vctk_wav48_silence_trimmed_000.tar.bz2
    clean_fullband/datasets_fullband.clean_fullband.vctk_wav48_silence_trimmed_001.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.vctk_wav48_silence_trimmed_002.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.vctk_wav48_silence_trimmed_003.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.vctk_wav48_silence_trimmed_004.tar.bz2
    noise_fullband/datasets_fullband.noise_fullband.audioset_000.tar.bz2
    noise_fullband/datasets_fullband.noise_fullband.audioset_001.tar.bz2
    noise_fullband/datasets_fullband.noise_fullband.audioset_002.tar.bz2
    noise_fullband/datasets_fullband.noise_fullband.freesound_000.tar.bz2
    noise_fullband/datasets_fullband.noise_fullband.freesound_001.tar.bz2
    datasets_fullband.impulse_responses_000.tar.bz2
)

###############################################################
# This data is identical to non-personalized track 4th DNS Challenge clean speech
# Recommend to re-download the data using this script

AZURE_URL="https://dns4public.blob.core.windows.net/dns4archive/datasets_fullband"
OUTPUT_PATH="./datasets_fullband"
EXTRACT_PATH="./dns_data"

# Configuration
DRY_RUN=${DRY_RUN:-true}  # Set to false to actually download
AUTO_EXTRACT=${AUTO_EXTRACT:-false}  # Set to true to auto-extract after download
DELETE_AFTER_EXTRACT=${DELETE_AFTER_EXTRACT:-false}  # Delete archive after successful extraction

# Check if required tools are installed
if ! command -v azcopy &> /dev/null; then
    echo "Error: azcopy is not installed"
    echo "Please install azcopy first. See: install_azcopy.sh"
    exit 1
fi

if [ "$AUTO_EXTRACT" = true ]; then
    if ! command -v pv &> /dev/null; then
        echo "Warning: pv is not installed. Install with: brew install pv (macOS) or apt install pv (Linux)"
        echo "Falling back to extraction without progress bar"
        USE_PV=false
    else
        USE_PV=true
    fi

    # Note: For bzip2 archives, we use pbzip2 (parallel bzip2) instead of pigz
    if ! command -v pbzip2 &> /dev/null; then
        echo "Warning: pbzip2 is not installed. Install with: brew install pbzip2 (macOS) or apt install pbzip2 (Linux)"
        echo "Falling back to single-threaded bzip2 decompression"
        USE_PIGZ=false
    else
        USE_PIGZ=true
    fi
fi

# Create output directories
mkdir -p "$OUTPUT_PATH/clean_fullband"
mkdir -p "$EXTRACT_PATH/dns_clean"
mkdir -p "$EXTRACT_PATH/dns_noise"

echo "=========================================="
echo "DNS4 Dataset Download using AzCopy"
echo "=========================================="
echo "Total files to download: ${#BLOB_NAMES[@]}"
echo "Download path: $OUTPUT_PATH"
echo "Extract path: $EXTRACT_PATH"
echo "DRY RUN: $DRY_RUN"
echo "Auto-extract: $AUTO_EXTRACT"
if [ "$AUTO_EXTRACT" = true ]; then
    echo "Use pv: $USE_PV"
    echo "Use pigz: $USE_PIGZ"
    echo "Delete after extract: $DELETE_AFTER_EXTRACT"
fi
echo "=========================================="
echo ""

# Initialize counters
TOTAL=${#BLOB_NAMES[@]}
CURRENT=0
SUCCESS=0
FAILED=0

for BLOB in "${BLOB_NAMES[@]}"; do
    CURRENT=$((CURRENT + 1))
    URL="$AZURE_URL/$BLOB"
    OUTPUT_FILE="$OUTPUT_PATH/$BLOB"

    # Get filename for display
    FILENAME=$(basename "$BLOB")

    echo "[$CURRENT/$TOTAL] Processing: $FILENAME"

    if [ "$DRY_RUN" = true ]; then
        # Dry run: just list the blob properties
        echo "  URL: $URL"
        echo "  Would download to: $OUTPUT_FILE"
        echo "  Status: DRY RUN (no actual download)"
        SUCCESS=$((SUCCESS + 1))
    else
        # Determine target directory based on filename (for skip check)
        if [[ "$FILENAME" =~ clean ]]; then
            TARGET_DIR="$EXTRACT_PATH/dns_clean"
        elif [[ "$FILENAME" =~ noise ]]; then
            TARGET_DIR="$EXTRACT_PATH/dns_noise"
        else
            TARGET_DIR="$EXTRACT_PATH/dns_clean"
        fi

        # Create subdirectory if needed
        mkdir -p "$(dirname "$OUTPUT_FILE")"
        mkdir -p "$TARGET_DIR"

        # Skip if archive already exists
        if [ -f "$OUTPUT_FILE" ]; then
            echo "  Status: Archive already exists, skipping..."
            SUCCESS=$((SUCCESS + 1))
        else
            # Download using azcopy
            if azcopy copy "$URL" "$OUTPUT_FILE" \
                --overwrite=false \
                --output-level=essential \
                --check-length=true; then
                echo "  Status: Download SUCCESS"
                SUCCESS=$((SUCCESS + 1))

                # Auto-extract if enabled
                if [ "$AUTO_EXTRACT" = true ]; then
                    echo "  Extracting to: $TARGET_DIR"

                    # Get file size for progress bar
                    FILE_SIZE=$(stat -f%z "$OUTPUT_FILE" 2>/dev/null || stat -c%s "$OUTPUT_FILE" 2>/dev/null)

                    # Extract with pv and pigz if available
                    EXTRACT_SUCCESS=false
                    if [ "$USE_PV" = true ] && [ "$USE_PIGZ" = true ]; then
                        # Most efficient: pv + pigz (parallel bzip2) + tar
                        # Note: pigz doesn't natively support bzip2, so we use pbzip2 if available
                        if command -v pbzip2 &> /dev/null; then
                            pv -s "$FILE_SIZE" "$OUTPUT_FILE" | pbzip2 -dc | tar -xC "$TARGET_DIR" && EXTRACT_SUCCESS=true
                        else
                            # Fallback to regular bzip2 with pv
                            pv -s "$FILE_SIZE" "$OUTPUT_FILE" | bzip2 -dc | tar -xC "$TARGET_DIR" && EXTRACT_SUCCESS=true
                        fi
                    elif [ "$USE_PV" = true ]; then
                        # pv only (no parallel decompression)
                        pv -s "$FILE_SIZE" "$OUTPUT_FILE" | bzip2 -dc | tar -xC "$TARGET_DIR" && EXTRACT_SUCCESS=true
                    else
                        # No progress bar, standard extraction
                        if command -v pbzip2 &> /dev/null; then
                            pbzip2 -dc "$OUTPUT_FILE" | tar -xC "$TARGET_DIR" && EXTRACT_SUCCESS=true
                        else
                            tar -xjf "$OUTPUT_FILE" -C "$TARGET_DIR" && EXTRACT_SUCCESS=true
                        fi
                    fi

                    if [ "$EXTRACT_SUCCESS" = true ]; then
                        echo "  Extraction: SUCCESS"
                        # Remove archive if requested
                        if [ "$DELETE_AFTER_EXTRACT" = true ]; then
                            rm "$OUTPUT_FILE"
                            echo "  Deleted archive: $FILENAME"
                        fi
                    else
                        echo "  Extraction: FAILED"
                        FAILED=$((FAILED + 1))
                    fi
                fi
            else
                echo "  Status: Download FAILED"
                FAILED=$((FAILED + 1))
            fi
        fi
    fi
    echo ""
done

echo "=========================================="
echo "Download Summary"
echo "=========================================="
echo "Total files: $TOTAL"
echo "Successful: $SUCCESS"
echo "Failed: $FAILED"
echo "=========================================="

if [ "$DRY_RUN" = true ]; then
    echo ""
    echo "This was a DRY RUN. To actually download files:"
    echo "  DRY_RUN=false bash $0"
    echo ""
    echo "To download and auto-extract:"
    echo "  DRY_RUN=false AUTO_EXTRACT=true bash $0"
    echo ""
    echo "To download, extract, and delete archives:"
    echo "  DRY_RUN=false AUTO_EXTRACT=true DELETE_AFTER_EXTRACT=true bash $0"
    echo ""
    echo "For best performance, install pv and pbzip2:"
    echo "  macOS: brew install pv pbzip2"
    echo "  Linux: apt install pv pbzip2"
fi

exit 0