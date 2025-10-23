#!/usr/bin/env bash
set -euo pipefail

# Optional: fetch the latest load_test.py from an external URL if provided
if [[ -n "${LOAD_TEST_SOURCE_URL:-}" ]]; then
  echo "Fetching latest load_test.py from $LOAD_TEST_SOURCE_URL"
  curl -fsSL "$LOAD_TEST_SOURCE_URL" -o /app/load_test.py
fi

# Check if Control API mode is enabled (default: false)
CONTROL_API_ENABLED="${CONTROL_API_ENABLED:-false}"

if [[ "$CONTROL_API_ENABLED" == "true" || "$CONTROL_API_ENABLED" == "1" ]]; then
  echo "Starting in Control API mode..."
  echo "Control API will be available on port ${CONTROL_API_PORT:-8080}"
  exec python /app/load_test.py
else
  # Original behavior: run load test with optional args
  PYTHON_CMD=(python /app/load_test.py)

  if [[ -n "${LOAD_TEST_EXTRA_ARGS:-}" ]]; then
    # shellcheck disable=SC2206
    EXTRA_ARGS=( ${LOAD_TEST_EXTRA_ARGS} )
    PYTHON_CMD+=("${EXTRA_ARGS[@]}")
  fi

  exec "${PYTHON_CMD[@]}"
fi
