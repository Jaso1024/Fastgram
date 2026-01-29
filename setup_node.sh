#!/bin/bash
set -e

echo ">>> Setting up Fastgram Build Node..."

# 1. Install System Dependencies
sudo apt-get update
sudo apt-get install -y cmake build-essential python3-pip git htop pv python3-venv

# 2. Clone Repository (if not exists) or Pull
if [ ! -d "Fastgram" ]; then
    echo ">>> Cloning Fastgram..."
    # Cloning from public repo for now, but usually we'd sync local changes.
    # Since I have unpushed local changes (the factory scripts), I need to upload them.
    # The user is running this script ON the node? No, I will run this via ssh.
    # So I will assume I rsync the files first.
    echo "Assuming Fastgram directory is synced."
else
    echo ">>> Fastgram directory exists."
fi

# 3. Install Python Deps
echo ">>> Installing Python dependencies..."
# Create venv to avoid conflicts
if [ -d "venv" ]; then
    echo "Removing broken/old venv..."
    rm -rf venv
fi
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install package
echo ">>> Installing Fastgram and requirements..."
pip install zstandard
pip install .
if [ -f "index_factory/requirements.txt" ]; then
    pip install -r index_factory/requirements.txt
fi

echo ">>> Setup Complete. Ready to build."
