#!/bin/bash

# Check if target folder is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <target_folder>"
  echo "Example: $0 /path/to/repository"
  exit 1
fi

# Variables
TARGET_FOLDER="$1"  # Target folder where the repository is located

# Navigate to the repository directory
cd "$TARGET_FOLDER" || { echo "Failed to navigate to repository directory: $TARGET_FOLDER"; exit 1; }

# Fetch the latest changes from the remote repository
echo "Fetching latest changes from origin..."
git fetch origin || { echo "Failed to fetch changes from origin"; exit 1; }

# Switch to the latest origin/main branch
echo "Switching to latest origin/main..."
git checkout origin/main || { echo "Failed to switch to origin/main"; exit 1; }

# Get the latest commit hash
LATEST_COMMIT=$(git rev-parse HEAD)

# Print the latest commit hash
echo "Successfully switched to latest origin/main."
echo "$LATEST_COMMIT"