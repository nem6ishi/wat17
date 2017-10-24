# Preprocessing scripts for WAT'16 en-ja tasks

## ASPEC en-ja translation
Set `ASPEC_JE_DIR` and `TARGET_DIR` correctly, then run:
```
$ ./ASPEC_JE_baseline.sh
```
The script will automatically do all the preprocessings [here](http://lotus.kuee.kyoto-u.ac.jp/WAT/baseline/dataPreparationJE.html) and place the files as:
```
$ tree $TARGET_DIR/corpus.baseline
corpus.baseline
├── dev
│   ├── dev.en
│   └── dev.ja
├── test
│   ├── test.en
│   └── test.ja
├── train
│   ├── train.en
│   └── train.ja
└── vocab
    ├── all.en
    └── all.ja
```

There are some differences from ASPEC official pre-processing procedure:
1. uses only train-1 and train-2 as training data
2. uses Kytea instead of Juman
3. uses remove-space.sh & h2z-utf8-without-space.pl following -> http://lotus.kuee.kyoto-u.ac.jp/WAT/evaluation/automatic_evaluation_systems/automaticEvaluationJA.html
4. limits max sentence length to 50 for both source and target
---

## Sentencepiece
After running `ASPEC_JE_baseline.sh`, run:
```
$ ./sentencepiece.sh $TARGET_DIR/corpus.baseline $VOCAB_SIZE
```
and get
```
$ tree $TARGET_DIR/corpus.baseline.sp${VOCAB_SIZE}
corpus.baseline.sp8000
├── dev
│   ├── dev.en
│   └── dev.ja
├── sp8000.model
├── sp8000.vocab
├── test
│   ├── test.en
│   └── test.ja
├── train
│   ├── train.en
│   └── train.ja
└── vocab
    ├── en_ja.8000
    └── en_ja.all
```
---

## Initialization with Word2vec
First, set `EMBEDDING_DIM` (default=512) in `init_vocab_w2v.sh`.
After running `sentencepiece.sh`, run:
```
$ ./init_vocab_w2v.sh $TARGET_DIR/corpus.baseline.sp${VOCAB_SIZE}/train \
  $TARGET_DIR/corpus.baseline.sp${VOCAB_SIZE}/vocab/en_ja.${VOCAB_SIZE}
```
and get `emb.npy` in the `vocab` directory.
This file can be fed into our seq2seq model by setting additional `model_params` as:
```
  --model_params "
      vocab_source: emb.npy
      vocab_target: emb.npy
      vocab_source_embedding: $EMB_SOURCE
      vocab_target_embedding: $EMB_TARGET" \
```
---

## Chunking with J.DepP
After running `ASPEC_JE_baseline.sh`, run:
```
$ ./ASPEC_JE_chunk.sh
```
and get following files:
```
$ tree $TARGET_DIR/corpus.chunk
corpus.chunk/
├── dev
│   ├── dev.en
│   └── dev.ja
├── test
│   ├── test.en
│   └── test.ja
├── train
│   ├── train.en
│   └── train.ja
└── vocab
    ├── all.en
    └── all.ja
```