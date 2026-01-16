#!/bin/bash
# Sync simple_checker to jetson-food-ai repo
# Usage: ./scripts/sync_to_jetson.sh

set -e

SRC_DIR="/home/youngjin/dku_frying_ai/src/simple_checker"
DST_DIR="/home/youngjin/jetson-food-ai/jetson2_frying_ai/simple_checker"
JETSON_REPO="/home/youngjin/jetson-food-ai"

echo "=== Sync simple_checker to jetson-food-ai ==="

# Check source exists
if [ ! -d "$SRC_DIR" ]; then
    echo "Error: Source directory not found: $SRC_DIR"
    exit 1
fi

# Create destination if not exists
mkdir -p "$DST_DIR"

# Copy files
echo "Copying files..."
cp -r "$SRC_DIR"/* "$DST_DIR"/

echo "Files synced:"
ls -la "$DST_DIR"

# Git commit & push
echo ""
echo "Committing to jetson-food-ai..."
cd "$JETSON_REPO"

git add jetson2_frying_ai/simple_checker/
git status --short

read -p "Commit and push? (y/n): " confirm
if [ "$confirm" = "y" ]; then
    git commit -m "Sync simple_checker from dku_frying_ai"
    git push
    echo ""
    echo "=== Done! ==="
    echo "Run on Jetson: cd ~/jetson-food-ai && git pull"
else
    echo "Skipped commit. Files are staged."
fi
