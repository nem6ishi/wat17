#!/bin/bash

if [ $# -ne 2 ]; then
    echo "USAGE: $0 train_dir vocab_file" 1>&2
    echo "E.g.,  $0 /your/train/directory /your/vocab/file"
    exit 1
fi

TRAIN_DIR=$1
VOCAB=$2

# When running this script, input your own script path below.
SCRIPT_DIR=/your/script/directory/init_vocab
PYTHON3=/your/python3/path/python3
W2V=/your/word2vec/bin/path/word2vec
EMBEDDING_DIM=256

cd $TRAIN_DIR
# Add tags and combine parallel corpora into a single file
$PYTHON3 $SCRIPT_DIR/add_tags.py < train.en > train.en_ja.tagged
$PYTHON3 $SCRIPT_DIR/add_tags.py < train.ja >> train.en_ja.tagged

# Run word2vec
$W2V -train train.en_ja.tagged -output emb.all -size $EMBEDDING_DIM -threads 8 -min-count 0

# Reshape embedding file and convert it to a numpy file
$PYTHON3 $SCRIPT_DIR/reshape_embedding.py -i emb.all -v $VOCAB -o emb.reshaped
$PYTHON3 $SCRIPT_DIR/save_embedding.py -i emb.reshaped -o emb_${EMBEDDING_DIM}.npy

mv $TRAIN_DIR/emb_${EMBEDDING_DIM}.npy $TRAIN_DIR/../vocab/
rm $TRAIN_DIR/train.en_ja.tagged 2>/dev/null
