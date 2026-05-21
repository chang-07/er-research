#!/usr/bin/env bash
# Retrain bi-encoder (v2_full) at multiple training-set sizes for the
# data-efficiency scaling curve. Sequential to avoid MPS contention.
#
# Output: data/predictions/method_v2_full_pct{10,25,50,75,100}.parquet
#
# Caller: bash src/method/run_data_efficiency.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

for pct in 10 25 50 75 100; do
  # 27,201 transitive-augmented training positives.
  # Cap at int(pct/100 * 27201).
  cap=$(python3 -c "print(int($pct / 100 * 27201))")
  echo ""
  echo "=== running pct=$pct (cap=$cap) ==="
  python3 src/method/train.py \
    --data data \
    --variant v2_full \
    --base-model intfloat/e5-small-v2 \
    --out-name method_v2_full_pct${pct} \
    --max-train-pairs $cap
done

echo "all data-efficiency runs complete"
