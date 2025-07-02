#!/bin/bash
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")
DATA_DIR="$PROJECT_ROOT/data/processed"

mkdir -p "$DATA_DIR"

handle_error() {
    echo "error: step $1 failed!"
    exit 1
}

echo "========================================="
echo "start"
echo "========================================="

GENERATE_SCRIPT="$SCRIPT_DIR/data_generation.py"
FILTER_SCRIPT="$SCRIPT_DIR/filter_answers.py"
INPUT_DATASET="hiyouga/math12k"
MODEL="DeepSeek-R1-Distill-Qwen-32B"
BATCH_SIZE=64
TENSOR_PARALLEL_SIZE=2
GENERATED_FILE="$DATA_DIR/result.json"
FILTERED_FILE="$DATA_DIR/filtered_data.json"

echo "[step 1/2]"
python3 $GENERATE_SCRIPT \
    --model $MODEL \
    --dataset $INPUT_DATASET \
    --batch_size $BATCH_SIZE \
    --tensor_parallel_size $TENSOR_PARALLEL_SIZE \
    --output $GENERATED_FILE

if [ $? -ne 0 ]; then
    handle_error 1
fi

echo "[step 1/2] Done!"
echo ""

echo "[step 2/2]"
python3 $FILTER_SCRIPT \
    --input $GENERATED_FILE \
    --output $FILTERED_FILE

if [ $? -ne 0 ]; then
    handle_error 2
fi

echo "[step 2/2] Done！"
echo ""

echo "========================================="
echo "Data Generation Path: ${GENERATED_FILE}"
echo "Filtered Outcome Path: ${FILTERED_FILE}"
echo "========================================="