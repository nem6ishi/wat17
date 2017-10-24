#!/bin/bash
# Ref: http://lotus.kuee.kyoto-u.ac.jp/WAT/baseline/dataPreparationJE.html

if [ $# -ne 2 ]; then
    echo "USAGE: $0 corpus_baseline_dir vocab_size" 1>&2
    echo "E.g.,  $0 /your/model/directory 8000"
    exit 1
fi

CORPUS_DIR=$1
VOCAB_SIZE=$2

# When running this script, input your own script path below.
SCRIPT_DIR=/your/script/directory
TARGET_DIR=$CORPUS_DIR.sp$VOCAB_SIZE

# Input path to your spm_train and spm_encode
SPM_TRAIN=/your/path/to/sentencepiece/src/spm_train
SPM_ENCODE=/your/path/to/sentencepiece/src/spm_encode

# Make directories
mkdir $TARGET_DIR 2>/dev/null
cd $TARGET_DIR
mkdir train dev test vocab 2>/dev/null

# Train and encode
$SPM_TRAIN --input=${CORPUS_DIR}/train/train.ja,${CORPUS_DIR}/train/train.en -model_prefix=sp${VOCAB_SIZE} --vocab_size=`expr $VOCAB_SIZE + 3`
for file in train dev test; do
    for lang in en ja; do
        $SPM_ENCODE --model=sp${VOCAB_SIZE}.model < ${CORPUS_DIR}/${file}/${file}.${lang} > ${TARGET_DIR}/${file}/${file}.${lang} &
    done
done; wait

# Make vocab file
cp train/train.ja train/train.tmp
cat train/train.en >> train/train.tmp
python3 $SCRIPT_DIR/make_vocab.py < train/train.tmp > vocab/en_ja.all
head -${VOCAB_SIZE} vocab/en_ja.all > vocab/en_ja.${VOCAB_SIZE}
rm train/train.tmp
