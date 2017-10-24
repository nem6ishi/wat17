# Usage: python make_vocab.py < corpus > corpus.vocab
import sys

word2count = {}
for line in sys.stdin:
    line = line.strip().split(' ')
    for word in line:
        if word in word2count:
            word2count[word] += 1
        else:
            word2count[word] = 1

sorted_w2c = sorted(word2count.items(), key=lambda x: x[1], reverse=True)
for word, _ in sorted_w2c:
    print(word)
