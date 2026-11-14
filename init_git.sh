#!/usr/bin/env bash
set -e

git init
git branch -M main
git add .
git commit -m "chore: initial project structure, data contracts, and role assignments"

# Create working branches for both team members
git branch feature/data-pipeline
git branch feature/ml-model

echo "Git repository initialized!"
echo "Branches created:"
echo "  - main"
echo "  - feature/data-pipeline (Member 1: Data & Signal Processing)"
echo "  - feature/ml-model      (Member 2: ML & Evaluation)"
