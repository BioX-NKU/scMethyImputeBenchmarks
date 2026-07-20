#Take 2i as an example
export CUDA_VISIBLE_DEVICES=0

ENV_PATH="anaconda3/envs/deepcpg/bin"
DATA_PATH="$ENV_PATH/data_2i"
MODEL_PATH="$ENV_PATH/model_2i"

for BATCH in $(seq 1 12); do
  echo "Processing 2i batch $BATCH"

  TRAIN_FILES=($(eval echo $DATA_PATH/2i_$BATCH/c{1,2,3,4,5}_*.h5)) # Modify according to actual circumstances
  VAL_FILES=($(eval echo $DATA_PATH/2i_$BATCH/c{11}_*.h5)) # Modify according to actual circumstances
  OUT_DIR="$MODEL_PATH/2i_$BATCH"

  mkdir -p "$OUT_DIR/eval/hdf"

  python $ENV_PATH/dcpg_train.py  \
    "${TRAIN_FILES[@]}" \
    --val_files "${VAL_FILES[@]}" \
    --dna_model CnnL2h128 \
    --cpg_model RnnL1 \
    --joint_model JointL2h512 \
    --nb_epoch 20 \
    --out_dir "$OUT_DIR/"

  python $ENV_PATH/dcpg_eval.py \
    $DATA_PATH/batch_Ser_$BATCH/*.h5 \
    --model_files "$OUT_DIR/model.json" "$OUT_DIR/model_weights_val.h5" \
    --out_data "$OUT_DIR/eval/data.h5" \
    --out_report "$OUT_DIR/eval/report.tsv"

  python $ENV_PATH/dcpg_eval_export.py \
    "$OUT_DIR/eval/data.h5" \
    -o "$OUT_DIR/eval/hdf" \
    -f hdf

done