#!/bin/bash

# Fix permissions for Conda ToS acceptance
CONDA_DIR="/Users/chenghengli/.conda"

# Create .conda if not exists
if [ ! -d "$CONDA_DIR" ]; then
    mkdir -p "$CONDA_DIR"
    echo "Created $CONDA_DIR"
fi

# Create tos directory
TOS_DIR="$CONDA_DIR/tos"
if [ ! -d "$TOS_DIR" ]; then
    mkdir -p "$TOS_DIR"
    echo "Created $TOS_DIR"
fi

# Change ownership (assuming current user is chenghengli)
chown -R chenghengli "$CONDA_DIR"

# Set permissions
chmod -R u+rw "$CONDA_DIR"

echo "Permissions fixed. Please try creating the environment again."
