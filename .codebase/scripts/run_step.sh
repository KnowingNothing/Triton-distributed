#!/bin/bash

# This wrapper script executes each step of the CI.
# It first checks the decision file; if the decision is SKIP, it skips execution.

set -e

# Path to the decision file
DECISION_FILE="ci_decision.txt"

# Check if the decision file exists
if [ ! -f "$DECISION_FILE" ]; then
  echo "Decision file '$DECISION_FILE' not found! The 'Pre-flight Check' step must run first."
  exit 1
fi

# Read the decision
DECISION=$(cat "$DECISION_FILE")

if [ "$DECISION" = "SKIP" ]; then
  echo "Skipping step because decision was SKIP."
  exit 0
else
  # If the decision is PROCEED, execute all commands passed as arguments
  echo "+++ Executing: $@"
  eval "$@"
fi