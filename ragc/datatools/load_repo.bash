#!/bin/bash

# Check if GitHub repository, target folder, and date are provided
if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ]; then
  echo "Usage: $0 <github_repo> <target_folder> <date> [branch]"
  echo "GitHub repo format: username/repo (e.g., PromtEngineer/localGPT)"
  echo "Date format: YYYY-MM-DD HH:MM:SS"
  echo "Example: $0 PromtEngineer/localGPT /path/to/folder '2023-05-28 14:37:54' main"
  exit 1
fi

# Variables
GITHUB_REPO="$1"  # GitHub repository in format username/repo
TARGET_FOLDER="$2"  # Target folder to clone the repository into
DATE="$3"          # Date passed as the third argument
BRANCH="${4:-main}"  # Branch passed as the fourth argument (default: main)

# Clone the repository into the target folder
echo "Cloning repository: $GITHUB_REPO into $TARGET_FOLDER..."
git clone "https://github.com/$GITHUB_REPO.git" "$TARGET_FOLDER" || { echo "Failed to clone repository"; exit 1; }

# Navigate to the repository directory
cd "$TARGET_FOLDER" || { echo "Failed to navigate to repository directory: $TARGET_FOLDER"; exit 1; }

# Pull the latest changes from the remote repository (in case of existing repo)
git pull origin "$BRANCH" || { echo "Failed to pull changes"; exit 1; }

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

echo "Successfully cloned repository and switched to commit: $COMMIT_HASH"
echo $COMMIT_HASH