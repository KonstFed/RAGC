#!/bin/bash

# Check if target folder and date are provided
if [ -z "$1" ] || [ -z "$2" ]; then
  echo "Usage: $0 <target_folder> <date>"
  echo "Date format: YYYY-MM-DD HH:MM:SS"
  echo "Example: $0 /path/to/folder '2023-05-28 14:37:54'"
  exit 1
fi

# Variables
TARGET_FOLDER="$1"  # Target folder where the repository is located
DATE="$2"          # Date passed as the second argument

# Navigate to the repository directory
cd "$TARGET_FOLDER" || { echo "Failed to navigate to repository directory: $TARGET_FOLDER"; exit 1; }

# Find the first commit before the specified date
COMMIT_INFO=$(git log --before="$DATE" --format=format:"%H %aI" | tail -n 1)

# Check if a commit was found
if [ -z "$COMMIT_INFO" ]; then
  echo "No commits found before $DATE"
  exit 1
fi

# Extract commit hash and date
COMMIT_HASH=$(echo "$COMMIT_INFO" | awk '{print $1}')
COMMIT_DATE=$(echo "$COMMIT_INFO" | awk '{print $2}')

# Print commit hash and date
echo "Commit Hash: $COMMIT_HASH"
echo "Commit Date: $COMMIT_DATE"

# Switch to the commit
git checkout "$COMMIT_HASH" || { echo "Failed to checkout commit"; exit 1; }

echo $COMMIT_HASH"