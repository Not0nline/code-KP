#!/usr/bin/env bash
set -euo pipefail

# Optional: fetch the latest load_test.py from an external URL if provided
if [[ -n "${LOAD_TEST_SOURCE_URL:-}" ]]; then
  echo "Fetching latest load_test.py from $LOAD_TEST_SOURCE_URL"
  curl -fsSL "$LOAD_TEST_SOURCE_URL" -o /app/load_test.py
fi

PYTHON_CMD=(python /app/load_test.py \
  --target "$TARGET" \
  --hpa-url "$HPA_URL" \
  --combined-url "$COMBINED_URL" \
  --predictive-url "$PREDICTIVE_URL" \
  --duration "$DURATION" \
  --max-concurrency "$MAX_CONCURRENCY" \
  --metrics-port "$METRICS_PORT" \
  --normal-min "$NORMAL_MIN" \
  --normal-max "$NORMAL_MAX" \
  --peak-min "$PEAK_MIN" \
  --peak-max "$PEAK_MAX" \
  --season-duration "$SEASON_DURATION" \
  --peak-duration "$PEAK_DURATION" \
  --peak-offset "$PEAK_OFFSET" \
  --volatility "$VOLATILITY" \
  --timeout "$TIMEOUT"
)

# Toggle predictive flag
if [[ "${TEST_PREDICTIVE,,}" == "true" ]]; then
  PYTHON_CMD+=(--test-predictive)
fi

exec "${PYTHON_CMD[@]}"
