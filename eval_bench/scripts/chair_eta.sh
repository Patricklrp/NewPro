#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash eval_bench/scripts/chair_eta.sh --log logs/chair_gpu3/llava-v1.5-7b/2/log.txt --out chair_results/llava-v1.5-7b_canny_gpu3/exp_002.jsonl --total 500
#   bash eval_bench/scripts/chair_eta.sh --log <log> --out <out> --total 500 --watch 10

LOG_FILE=""
OUT_FILE=""
TOTAL="500"
WATCH_SEC="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --log)
      LOG_FILE="$2"
      shift 2
      ;;
    --out)
      OUT_FILE="$2"
      shift 2
      ;;
    --total)
      TOTAL="$2"
      shift 2
      ;;
    --watch)
      WATCH_SEC="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "$LOG_FILE" || -z "$OUT_FILE" ]]; then
  echo "Usage: $0 --log <log_file> --out <jsonl_file> [--total 500] [--watch 10]" >&2
  exit 1
fi

print_eta_once() {
  if [[ ! -f "$LOG_FILE" ]]; then
    echo "log file not found: $LOG_FILE"
    return 1
  fi

  local done=0
  if [[ -f "$OUT_FILE" ]]; then
    done=$(wc -l < "$OUT_FILE" | tr -d ' ')
  fi

  local start
  start=$(head -n 1 "$LOG_FILE" | sed -n 's/^\[\([^]]*\)\].*$/\1/p')

  if [[ -z "$start" ]]; then
    echo "Cannot parse start timestamp from: $LOG_FILE"
    return 1
  fi

  local now
  now=$(date '+%Y-%m-%d %H:%M:%S')

  local s_ts n_ts
  s_ts=$(date -d "$start" +%s)
  n_ts=$(date -d "$now" +%s)

  local elapsed=$((n_ts - s_ts))
  if [[ "$elapsed" -lt 1 ]]; then
    elapsed=1
  fi

  local remain=$((TOTAL - done))
  if [[ "$remain" -lt 0 ]]; then
    remain=0
  fi

  local avg eta
  if [[ "$done" -gt 0 ]]; then
    avg=$(awk -v e="$elapsed" -v d="$done" 'BEGIN{printf "%.4f", e/d}')
    eta=$(awk -v r="$remain" -v a="$avg" 'BEGIN{printf "%d", r*a}')
  else
    avg="nan"
    eta=0
  fi

  local eta_h eta_m eta_s
  eta_h=$((eta / 3600))
  eta_m=$(((eta % 3600) / 60))
  eta_s=$((eta % 60))

  printf '[ETA] done=%d/%d remain=%d elapsed=%ds avg=%ss eta=%02d:%02d:%02d now=%s\n' \
    "$done" "$TOTAL" "$remain" "$elapsed" "$avg" "$eta_h" "$eta_m" "$eta_s" "$now"
}

if [[ "$WATCH_SEC" -gt 0 ]]; then
  while true; do
    print_eta_once || true
    sleep "$WATCH_SEC"
  done
else
  print_eta_once
fi
