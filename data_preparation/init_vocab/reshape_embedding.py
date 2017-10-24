#!/usr/bin/env python
#coding:utf-8

import numpy as np
import sys, time
import argparse

parser = argparse.ArgumentParser(description='Reshape the word init_vocab using reference vocabrary.')
parser.add_argument('-i', '--input', nargs='?', required=True)
parser.add_argument('-v', '--vocab', nargs='?', required=True)
parser.add_argument('-o', '--output', nargs='?', required=True)
args = parser.parse_args()


#make embedding_dict
sys.stderr.write('Reading word init_vocab data...\n');sys.stderr.flush()
f = open(args.input, 'r')
embedding_data = f.readlines()
f.close()
dim = int(embedding_data[0].split()[1])
embedding_dict = {}
for each in embedding_data[1:]:
    each_list = each.split()
    embedding_dict[each_list[0]] = list(map(float, each_list[1:]))

#make vocab_list
sys.stderr.write('Reading vocab data...\n');sys.stderr.flush()
vocab_list = []
for line in open(args.vocab, 'r'):
    vocab_list.append(line.strip().split('\t', 1)[0])
vocab_list.extend(['UNK', 'SEQUENCE_START', 'SEQUENCE_END'])


#make 'UNK' embedding using average of out-of-vocab words
sys.stderr.write('Making unk init_vocab...\n');sys.stderr.flush()
count = 0
end = len(embedding_dict.keys())
average = np.zeros(dim)
for i, each in enumerate(embedding_dict.keys()):
    if each not in vocab_list:
        count += 1
        average += embedding_dict[each]
average /= count
embedding_dict['UNK'] = average.tolist()

#output as 'word value, , , value \n'
sys.stderr.write('Writing the result...');sys.stderr.flush()
f = open(args.output, 'w')
output_list = []
end = len(vocab_list)
unk_count = 0
for i, each in enumerate(vocab_list):
    if each in embedding_dict:
        f.write(each + " " + ' '.join(map(str, embedding_dict[each])) + '\n')
    else:
        f.write(each + " " + ' '.join(map(str, embedding_dict['UNK'])) + '\n')
        unk_count += 1
f.close()
sys.stderr.write('\n--------------------\n # of UNK: ' + str(unk_count) + '\n--------------------')
