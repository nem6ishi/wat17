#!/usr/bin/env python
#coding:utf-8

import sys
for line in sys.stdin:
  print("SEQUENCE_START", line.strip('\n'), "SEQUENCE_END")
