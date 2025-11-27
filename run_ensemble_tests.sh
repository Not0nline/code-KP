#!/bin/bash

# ENSEMBLE TEST RUNNER
# Usage: ./run_ensemble_tests.sh [SUDO_PREFIX]

SUDO_CMD=$1

echo "=== STARTING ENSEMBLE TEST SUITE ==="
echo "This script will run the ensemble tests (Low, Medium, High) using the cluster orchestrator."

# Ensure the orchestrator script exists and is executable
if [ ! -f "./cluster_orchestrator.sh" ]; then
    echo "Error: cluster_orchestrator.sh not found!"
    exit 1
fi

chmod +x ./cluster_orchestrator.sh

# Run the orchestrator in 'ensemble' mode
# This will iterate through low, medium, and high scenarios for the ensemble model
nohup ./cluster_orchestrator.sh ensemble $SUDO_CMD > ensemble_orchestrator.log 2>&1 &

echo "Ensemble tests started in background."
echo "Logs are being written to ensemble_orchestrator.log"
echo "You can check progress with: tail -f ensemble_orchestrator.log"
