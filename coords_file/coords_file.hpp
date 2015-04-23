/*
Copyright (c) 2015, Florian Sittel (www.lettis.net)
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <memory>

extern "C" {
  // use xdrfile library to read/write xtc and trr files from gromacs
  #include "xdrfile/xdrfile.h"
  #include "xdrfile/xdrfile_xtc.h"
  #include "xdrfile/xdrfile_trr.h"
}

namespace CoordsFile {

class Handler {
 public:
  virtual std::vector<float> next() = 0;
  virtual void write(std::vector<float> row) = 0;
  virtual bool eof() = 0;
};

class AsciiHandler : public Handler {
 public:
  AsciiHandler(std::string fname, std::string mode);
  std::vector<float> next();
  void write(std::vector<float> row);
  bool eof();
 protected:
  std::ifstream _ifs;
  std::ofstream _ofs;
  bool _eof;
  std::string _mode;
};

class XtcHandler : public Handler {
 public:
  XtcHandler(std::string fname, std::string mode);
  ~XtcHandler();
  std::vector<float> next();
  void write(std::vector<float> row);
  bool eof();
 protected:
  bool _eof;
  std::string _mode;
  int _natoms;
  int _nrow;
  XDRFILE* _xdr;
  rvec* _coord_buf;
};

typedef std::unique_ptr<Handler> FilePointer;

template <typename T> std::vector<T>
split(std::string s);

FilePointer
open(std::string fname, std::string mode);

} // end namespace 'CoordsFile'

