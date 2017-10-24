DATA_PATH=./data
VOCAB=$DATA_PATH/vocab
VOCAB_EMBEDDING=$DATA_PATH/vocab_embedding.npy
TRAIN_SOURCE=$DATA_PATH/train/en
TRAIN_TARGET=$DATA_PATH/train/ja
DEV_SOURCE=$DATA_PATH/dev/en
DEV_TARGET=$DATA_PATH/dev/ja

BATCHSIZE=256
TRAIN_STEPS=200000
EVAL_EVERY_N_STEPS=2000

MODEL_DIR=./model

python -m bin.train \
    --config_paths "
        ./example_configs/large_len50_lr00001_hl256.yml,
        ./example_configs/train_seq2seq.yml,
        ./example_configs/text_metrics_sp.yml" \
    --model_params "
        vocab_source: $VOCAB
        vocab_target: $VOCAB
        vocab_source_embedding: $VOCAB_EMBEDDING
        vocab_target_embedding: $VOCAB_EMBEDDING" \
    --input_pipeline_train "
          source_files:
            - $TRAIN_SOURCE
          target_files:
            - $TRAIN_TARGET" \
    --input_pipeline_dev "
        class: ParallelTextInputPipeline
        params:
          source_files:
            - $DEV_SOURCE
          target_files:
            - $DEV_TARGET" \
    --batch_size $BATCHSIZE \
    --train_steps $TRAIN_STEPS \
    --eval_every_n_steps $EVAL_EVERY_N_STEPS \
    --output_dir $MODEL_DIR \
    --keep_checkpoint_max 0 \
    --save_checkpoints_steps $EVAL_EVERY_N_STEPS \
    --gpu_allow_growth True > $MODEL_DIR/log.txt
