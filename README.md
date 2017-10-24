# WAT 2017: UT-IIS

This is the official code used for WAT 2017 Description Paper titled **A Bag of Useful Tricks for Practical Neural Machine Translation: Embedding Layer Initialization and Large Batch Size**.

This program is based on [Google's seq2seq implementation](https://github.com/google/seq2seq).

You may read the documentation of the original implementation at https://google.github.io/seq2seq/.

## Added Features

The following two features are added to the original implementation.
1. Embedding Layer Initialization
2. Ensemble of models

The example usage of our code are in `./examples` directory.

## Preparing Word Embedding File

To create word embedding, please refer to `./data_preparation/init_vocab_w2v.sh`.
This script train word embeddings and convert them into `.npy` format.


## Reference

```
@InProceedings{neishi:WAT2017,
  author    = {Neishi, Masato  and  Sakuma, Jin  and  Tohda, Satoshi  and  Ishiwatari, Shonosuke and Yoshinaga, Naoki and Toyoda, Masashi},
  title     = {A Bag of Useful Tricks for Practical Neural Machine Translation: Embedding Layer Initialization and Large Batch Size},
  booktitle = {Proceedings of the 4rd Workshop on Asian Translation (WAT2017)},
  year      = {2017 (to appear)}
}
```

The citation for Google's original paper.

```
@ARTICLE{Britz:2017,
  author          = {{Britz}, Denny and {Goldie}, Anna and {Luong}, Thang and {Le}, Quoc},
  title           = "{Massive Exploration of Neural Machine Translation Architectures}",
  journal         = {ArXiv e-prints},
  archivePrefix   = "arXiv",
  eprinttype      = {arxiv},
  eprint          = {1703.03906},
  primaryClass    = "cs.CL",
  keywords        = {Computer Science - Computation and Language},
  year            = 2017,
  month           = mar,
}
```
