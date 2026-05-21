#!/usr/bin/env bash
# Chain remaining compute jobs sequentially after CE-save completes.
# Order: score WA with v2-CE -> train MiniLM-L-12 -> score WA with L-12
#        -> run data efficiency (5 bi-encoder retrains) -> evaluate
#
# Caller: bash src/method/chained_compute.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

CE_MODEL=data/models/method_v2_crossenc_saved
WA_TEST=/tmp/magellan/walmart_amazon_test.txt
WA_TRAIN=/tmp/magellan/walmart_amazon_train.txt

# 1. Verify CE checkpoint
if [ ! -d "$CE_MODEL" ]; then
  echo "ERROR: $CE_MODEL does not exist; CE retrain did not save the model"
  exit 1
fi

# 2. Score WA with the saved v2 CE
echo "=== [1/5] scoring Walmart-Amazon with v2 CE (saved checkpoint) ==="
python3 src/method/score_external.py \
  --model-dir "$CE_MODEL" \
  --test-file "$WA_TEST" \
  --train-file "$WA_TRAIN" \
  --out data/predictions/method_v2_crossenc_on_walmart_amazon.parquet

# 3. Also baseline: E5 zero-shot on WA
echo "=== [2/5] scoring Walmart-Amazon with E5 zero-shot ==="
python3 src/method/score_external_e5.py \
  --test-file "$WA_TEST" \
  --train-file "$WA_TRAIN" \
  --out data/predictions/e5_zeroshot_on_walmart_amazon.parquet

# 4. Train MiniLM-L-12 cross-encoder (larger backbone variant)
echo "=== [3/5] training MiniLM-L-12 cross-encoder ==="
python3 src/method/train_cross_encoder.py \
  --data data \
  --hard-negs hard_negatives_v2.parquet \
  --train-pairs pairs.parquet \
  --base-model cross-encoder/ms-marco-MiniLM-L-12-v2 \
  --out-name method_v2_crossenc_L12

# 5. Score WA with L-12 CE (transfer + larger-backbone double-check)
if [ -d "data/models/method_v2_crossenc_L12" ]; then
  echo "=== [4/5] scoring Walmart-Amazon with L-12 CE ==="
  python3 src/method/score_external.py \
    --model-dir data/models/method_v2_crossenc_L12 \
    --test-file "$WA_TEST" \
    --train-file "$WA_TRAIN" \
    --out data/predictions/method_v2_crossenc_L12_on_walmart_amazon.parquet
else
  echo "WARN: L-12 CE save did not produce checkpoint dir; skipping WA-L12"
fi

# 6. Data-efficiency runs (5 bi-encoder retrains at 10/25/50/75/100% data)
echo "=== [5/5] running data-efficiency curve ==="
bash src/method/run_data_efficiency.sh

echo ""
echo "ALL COMPUTE COMPLETE"
