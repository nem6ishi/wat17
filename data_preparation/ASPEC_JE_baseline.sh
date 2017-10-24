#!/bin/bash
# Ref: http://lotus.kuee.kyoto-u.ac.jp/WAT/baseline/dataPreparationJE.html
# This script includes some differences from ASPEC official pre-processing procedure. It
# 1. uses only train-1 and train-2 as training data
# 2. uses Kytea instead of Juman
# 3. uses remove-space.sh & h2z-utf8-without-space.pl following -> http://lotus.kuee.kyoto-u.ac.jp/WAT/evaluation/automatic_evaluation_systems/automaticEvaluationJA.html
# 4. limits max sentence length to 50 for both source and target

# When running this script, input your own script path below.
ASPEC_JE_DIR=/path/to/ASPEC-JE/corpus
TARGET_DIR=$ASPEC_JE_DIR/wat17
MAX_SEN_LEN=50

SCRIPT_DIR=/your/script/directory/data_preparation
MOSES_SCRIPT=$SCRIPT_DIR/mosesdecoder-RELEASE-2.1.1/scripts
WAT_SCRIPT=$SCRIPT_DIR/script.segmentation.distribution
KYTEA=/your/directory/local/bin/kytea
KYTEA_MODEL=/your/kytea/model/directory/jp-0.4.2-utf8-1.mod

# Download Moses-2.1.1 & scripts from ASPEC page
cd $SCRIPT_DIR
if [ ! -d mosesdecoder-RELEASE-2.1.1 ]
then
    wget https://github.com/moses-smt/mosesdecoder/archive/RELEASE-2.1.1.zip
    unzip RELEASE-2.1.1
fi
if [ ! -d script.segmentation.distribution ]
then
    wget wget http://lotus.kuee.kyoto-u.ac.jp/WAT/evaluation/automatic_evaluation_systems/script.segmentation.distribution.tar.gz
    tar zxvf script.segmentation.distribution.tar.gz
fi
rm RELEASE-2.1.1.zip script.segmentation.distribution.tar.gz 2>/dev/null

# Copy data into corpus.org
mkdir $TARGET_DIR $TARGET_DIR/corpus.org 2>/dev/null
cd $TARGET_DIR
cp $ASPEC_JE_DIR/train/* corpus.org
cp $ASPEC_JE_DIR/dev/* corpus.org
cp $ASPEC_JE_DIR/test/* corpus.org

# Extract sentences
cd $TARGET_DIR/corpus.org/
for name in dev test; do
  perl -ne 'chomp; @a=split/ \|\|\| /; print $a[2], "\n";' < ${name}.txt > ${name}.ja.txt &
  perl -ne 'chomp; @a=split/ \|\|\| /; print $a[3], "\n";' < ${name}.txt > ${name}.en.txt &
done
for name in train-1 train-2 train-3; do
  perl -ne 'chomp; @a=split/ \|\|\| /; print $a[3], "\n";' < ${name}.txt > ${name}.ja.txt &
  perl -ne 'chomp; @a=split/ \|\|\| /; print $a[4], "\n";' < ${name}.txt > ${name}.en.txt &
done; wait

# Tokenize Japanese data
cd $TARGET_DIR
mkdir corpus.tok
cd corpus.tok
for file in train-1 train-2 dev test; do
  cat ../corpus.org/${file}.ja.txt | \
    perl -Mencoding=utf8 -pe 's/(.)［[０-９．]+］$/${1}/;' | \
    $WAT_SCRIPT/remove-space.sh | \
    perl ${WAT_SCRIPT}/h2z-utf8-without-space.pl | \
    $KYTEA -model $KYTEA_MODEL -out tok | \
    perl -Mencoding=utf8 -pe 'while(s/([０-９]) ([０-９])/$1$2/g){} s/([０-９]) (．) ([０-９])/$1$2$3/g; while(s/([Ａ-Ｚ]) ([Ａ-Ｚａ-ｚ])/$1$2/g){} while(s/([ａ-ｚ]) ([ａ-ｚ])/$1$2/g){}' \
    > ${file}.ja &
done; wait

# Tokenize English data
for file in train-1 train-2 train-3 dev test; do
  cat ../corpus.org/${file}.en.txt | \
    perl ${WAT_SCRIPT}/z2h-utf8.pl | \
    perl ${MOSES_SCRIPT}/tokenizer/tokenizer.perl -l en \
    > ${file}.tok.en &
done; wait

# Train truecaser for English
cat train-1.tok.en train-2.tok.en train-3.tok.en dev.tok.en > train_dev.tok.en
${MOSES_SCRIPT}/recaser/train-truecaser.perl --model truecase-model.en --corpus train_dev.tok.en

# Truecase English data
for file in train-1 train-2  dev test; do
  ${MOSES_SCRIPT}/recaser/truecase.perl --model truecase-model.en < ${file}.tok.en > ${file}.en &
done; wait

# Clean training data
cd $TARGET_DIR/corpus.tok
for file in train-1 train-2 ; do 
  perl ${MOSES_SCRIPT}/training/clean-corpus-n.perl ${file} ja en ${file}_cleaned 1 $MAX_SEN_LEN &
done; wait
cd $TARGET_DIR
mkdir corpus.baseline corpus.baseline/train corpus.baseline/dev corpus.baseline/test corpus.baseline/vocab
cd $TARGET_DIR/corpus.tok
cat train-1_cleaned.ja train-2_cleaned.ja > $TARGET_DIR/corpus.baseline/train/train.ja &
cat train-1_cleaned.en train-2_cleaned.en > $TARGET_DIR/corpus.baseline/train/train.en &
wait

# Copy dev and test data
for file in dev test; do
    cp $TARGET_DIR/corpus.tok/${file}.en $TARGET_DIR/corpus.baseline/${file}/${file}.en &
    cp $TARGET_DIR/corpus.tok/${file}.ja $TARGET_DIR/corpus.baseline/${file}/${file}.ja &
done; wait


# Make vocab file
for lang in ja en; do
    python $SCRIPT_DIR/make_vocab.py < $TARGET_DIR/corpus.baseline/train/train.$lang > $TARGET_DIR/corpus.baseline/vocab/all.$lang &
done; wait
