#!/usr/bin/env python
#coding:utf-8

import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Reshape the word init_vocab using reference vocabrary.')
parser.add_argument('-i', '--input', nargs='?', required=True)
parser.add_argument('-o', '--output', nargs='?', required=True)
args = parser.parse_args()

v = []
vocab_list = []
with open(args.input) as file:
    v = list(list(map(float, line.strip("\n").split()[1:])) for line in file)

np.save(args.output, v)
