#!/bin/bash

# Check if GitHub repository and target folder are provided
if [ -z "$1" ] || [ -z "$2" ]; then
  echo "Usage: $0 <github_repo> <target_folder> [branch]"
  echo "GitHub repo format: username/repo (e.g., PromtEngineer/localGPT)"
  echo "Example: $0 PromtEngineer/localGPT /path/to/folder main"
  exit 1
fi

# Variables
GITHUB_REPO="$1"  # GitHub repository in format username/repo
TARGET_FOLDER="$2"  # Target folder to clone the repository into
BRANCH="${3:-main}"  # Branch passed as the third argument (default: main)

# Clone the repository into the target folder
echo "Cloning repository: $GITHUB_REPO into $TARGET_FOLDER..."
git clone "git@github.com:$GITHUB_REPO.git" "$TARGET_FOLDER" || { echo "Failed to clone repository"; exit 1; }

# Navigate to the repository directory
cd "$TARGET_FOLDER" || { echo "Failed to navigate to repository directory: $TARGET_FOLDER"; exit 1; }

# Pull the latest changes from the remote repository (in case of existing repo)
git pull origin "$BRANCH" || { echo "Failed to pull changes"; exit 1; }

CURRENT_COMMIT=$(git rev-parse HEAD)

echo "Successfully cloned repository and pulled latest changes from branch: $BRANCH"
echo $CURRENT_COMMIT
