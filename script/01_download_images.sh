#!/bin/bash
# Step 1: Download BEV satellite images using Google Maps Static API
# Prerequisites: Google Maps API key and URL signing secret

IDEPS_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

python "$IDEPS_ROOT/source/bev_image_parser/bev_image_parser.py" \
    --coordinates_file "$IDEPS_ROOT/data/sample_coordinates.tsv" \
    --api_key_file "<PATH_TO_YOUR_API_KEY_FILE>" \
    --signature_key_file "<PATH_TO_YOUR_SIGNATURE_KEY_FILE>" \
    --output_dir "$IDEPS_ROOT/data/bev_images/" \
    --zoom_levels 15,17
