MODEL1_DIR=./model1
CKPT1=10000
MODEL2_DIR=./model2
CKPT2=10000
MODEL3_DIR=./model3
CKPT3=10000

DATA_PATH=./data
VOCAB=$DATA_PATH/vocab
SOURCE=$DATA_PATH/test/en

BEAM_WIDTH=16
LENGTH_PENALTY=1.0
BATCHSIZE=256

OUTPUT=./prediction/ja

python -m bin.ensemble\
    --tasks "
        - class: DecodeText
          params:
            postproc_fn: seq2seq.data.postproc.decode_sentencepiece" \
    --model_params "
        vocab_source: $VOCAB
        vocab_target: $VOCAB
        inference.beam_search.beam_width: $BEAM_WIDTH
        inference.beam_search.length_penalty_weight: $LENGTH_PENALTY" \
    --models "
        - dir: $MODEL1_DIR
          checkpoint_path: $CKPT1
        - dir: $MODEL2_DIR
          checkpoint_path: $CKPT2
        - dir: $MODEL3_DIR
          checkpoint_path: $CKPT3" \
    --input_pipeline "
        class: ParallelTextInputPipeline
        params:
          source_files:
            - $SOURCE" \
    --batchsize $BATCHSIZE > $OUTPUT
