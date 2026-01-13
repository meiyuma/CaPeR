#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

if [ -z "$1" ]; then
  echo "Error: Specify task name (e.g., ade, comagc, gene)"
  exit 1
fi
task=$1

train_path="${BASE_DIR}/data/${task}/train_full.jsonl"
test_path="${BASE_DIR}/data/${task}/test_truth.jsonl"
ckpt_dir="${BASE_DIR}/src/ptc/checkpoints/${task}"


echo ">>> [1/3] Running XGBoost Ranker..."
python src/ltr_xgboost.py \
  --task ${task} \
  --train_path ${train_path} \
  --test_path ${test_path} \
  --checkpoint_path ${ckpt_dir}/xgb \
  --epoch 5

echo ">>> [2/3] Running ListNet Ranker..."

python src/ltr_nn.py \
  --do_train True \
  --loss_type list_net \
  --model_name "models/roberta-base" \
  --train_path ${train_path} \
  --test_path ${test_path} \
  --checkpoint_path ${ckpt_dir}/nn \
  --epoch 5 \
  --do_eval True


best_file=$(ls -t ${ckpt_dir}/nn/test_pred_ep*.jsonl | head -1)
echo "Using ListNet output: ${best_file}"




echo ">>> [2.5/3] Running RankNet..."
python src/ltr_nn.py \
  --do_train True \
  --loss_type rank_net \
  --model_name "models/roberta-base" \
  --train_path ${train_path} \
  --test_path ${test_path} \
  --checkpoint_path ${ckpt_dir}/ranknet \
  --epoch 5 \
  --do_eval True

ranknet_file=$(ls -t ${ckpt_dir}/ranknet/test_pred_ep*.jsonl | head -1)


echo ">>> [3/3] Running RRF Ensemble (3 Models)..."
python src/ltr_ensemble.py \
  --input_files \
    "${ckpt_dir}/xgb/test_pred_xgboost.jsonl" \
    "${best_file}" \
    "${ranknet_file}" \
  --output_file "${ckpt_dir}/test_pred_ensemble.jsonl" \
  --top_k 5

echo "✅ All done! Ensemble Output: ${ckpt_dir}/test_pred_ensemble.jsonl"
