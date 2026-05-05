#!/bin/bash
set -e

# Move to the script's directory
cd "$(dirname "$0")"

echo "Syncing isolated MQT Bench environment..."

# uv sync creates the venv and resolves dependencies based on pyproject.toml
uv sync

echo "------------------------------------------"
echo "Setup complete! MQT Bench is ready."
echo "Dependencies are locked in $(pwd)/uv.lock"
echo "------------------------------------------"