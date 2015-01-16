#!/usr/bin/env python

import sys
import os

fh_in = open(sys.argv[1], 'r')
fh_out = open(sys.argv[2], 'w')

for line in fh_in:
  fh_out.write(r'"' + line[:-1] + r'\n"' + "\n")

fh_in.close()
fh_out.close()

