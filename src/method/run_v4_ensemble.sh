#!/usr/bin/env bash
# Chain v4 cross-encoder retrains with different seeds, then ensemble.
# Waits for the seed=42 run (started separately) to finish, then fires
# seed=7 and seed=1337 sequentially. ~17 min per seed.
#
# Caller: bash src/method/run_v4_ensemble.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

# Wait for the seed=42 model checkpoint to land.
echo "=== waiting for seed=42 enriched CE ==="
until [ -f data/models/method_v4_crossenc_enriched/config.json ] \
   && [ -f data/predictions/method_v4_crossenc_enriched.parquet ]; do
  sleep 15
done
echo "seed=42 done; starting seed=7"

python3 src/method/train_cross_encoder.py \
  --data data \
  --hard-negs hard_negatives_v2.parquet \
  --train-pairs pairs.parquet \
  --out-name method_v4_crossenc_enriched_s7 \
  --seed 7

echo ""
echo "seed=7 done; starting seed=1337"

python3 src/method/train_cross_encoder.py \
  --data data \
  --hard-negs hard_negatives_v2.parquet \
  --train-pairs pairs.parquet \
  --out-name method_v4_crossenc_enriched_s1337 \
  --seed 1337

echo ""
echo "all 3 seeds complete"
