
# Copyright (c) 2014, Florian Sittel
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met: 
# 
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer. 
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution. 
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.



# NOTE:  this module is written for python 2.7x


"""
md_toolkit: collection of MD-related python functions.
"""

import os
import tempfile
import random
import re

from multiprocessing import Process, Pipe
from itertools import izip

### MD-specific
def readIndices(filename):
  """read indices from a GROMACS index file.
     returns a dictionary with group names as keys
     and lists of indices as according values.
     e.g. {'System': [1,2,3,4,...], 'Backbone': [1,3,...], ...}"""
  group_name = None
  groups = {}
  fh = open(filename, 'r')
  group_def = re.compile("\[.*\]")
  group_name_def = re.compile("(\w|-)+")
  atom_def = re.compile("(\d+\s*)+")
  try:
    for line in fh.readlines():
      line = line.strip()
      if group_def.match(line):
        group_name = group_name_def.search(line).group(0)
        groups[group_name] = []
      else:
        if atom_def.match(line):
          groups[group_name].extend(map(int, line.split()))
  finally:
    fh.close()
  return groups

def writeIndices(filename, indices):
  """write indices to file (overwriting previously existing content).
     indices should be a dictionary with group definitions as keys
     and lists of indices as values.
     e.g. {'System': [1,2,3,4,...], 'Backbone': [1,3,...], ...}"""
  fh = open(filename, 'w')
  try:
    for group in indices:
      fh.write("[ %s ]\n" % group)
      fh.write("%s\n\n" % " ".join(map(str, indices[group])))
  finally:
    fh.close()

### general tools

def generateTempFilename(dir=".", ext=""):
  """generate name for temporary file.
     'dir' is the path, in which the file will be written (default: local path).
     'ext' is the filename suffix (with dot)
     (e.g. '.csv' for comma-separated value files).

     if 'dir' is given, this function will check,
     if a file under the generated name already exists and
     will choose another filename.
     HOWEVER: if another program writes to a file under the same
              name after the check has been done, read/write clashes
              may occur!"""
  tmp_prefix = tempfile.gettempprefix()
  if not dir[:-1] == "/":
    dir += "/"
  file_exists = True
  while (file_exists):
    rnd = random.randint(100000,999999)
    tmp_name = "%s%s%s%s" % (dir, tmp_prefix, rnd, ext)
    file_exists = os.path.isfile(tmp_name)
  return tmp_name

def parmap(f, X, n_threads=2):
  """run function 'f' in elements of list 'X' in parallel.
     returns list of results: [f(x_1), f(x_2), ..., f(x_N)].
     'n_threads' defines number of threads (default: 2)."""
  def spawn(f):
    def fun(pipe,x):
      pipe.send(f(x))
      pipe.close()
    return fun
  n_x = len(X)
  result = []
  for chunk in range(n_x/n_threads):
    XX = X[chunk*n_threads:(chunk+1)*n_threads]
    pipe=[Pipe() for x in XX]
    proc=[Process(target=spawn(f),args=(c,x)) for x,(p,c) in izip(XX,pipe)]
    [p.start() for p in proc]
    [p.join() for p in proc]
    result.extend([p.recv() for (p,c) in pipe])
  return result

