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
#include "coords_file.hpp"

#include <iostream>

namespace CoordsFile {

template <typename T>
std::vector<T>
split(std::string s) {
  std::vector<T> v;
  std::istringstream iss(s);
  while(iss.good()) {
    T buf;
    iss >> buf;
    v.push_back(buf);
  }
  return v;
}

//// ASCII handler

AsciiHandler::AsciiHandler(std::string fname, std::string mode)
  : _eof(false)
  , _mode(mode) {
  if (mode == "r") {
    this->_ifs.open(fname);
  } else if (mode == "w") {
    this->_ofs.open(fname);
  } else {
    std::cerr << "unknown mode: " << mode << std::endl;
    exit(EXIT_FAILURE);
  }
}

std::vector<float>
AsciiHandler::next() {
  if (_ifs.is_open() && _ifs.good()) {
    std::string s;
    std::getline(_ifs, s);
    if (_ifs.good()) {
      if (s == "") {
        // skip empty lines
        return this->next();
      } else {
        return split<float>(s);
      }
    }
  }
  _eof = true;
  return {};
}

void
AsciiHandler::write(std::vector<float> row) {
  if (_ofs.is_open() && _ofs.good()) {
    for (float f: row) {
      _ofs << " " << f;
    }
    _ofs << std::endl;
  }
}

bool
AsciiHandler::eof() {
  return _eof;
}


//// XTC handler

XtcHandler::XtcHandler(std::string fname, std::string mode)
  : _eof(false)
  , _mode(mode)
  , _nrow(0) {
  if (_mode == "r") {
    read_xtc_natoms(fname.c_str(), &_natoms);
    _coord_buf = static_cast<rvec*>(calloc(_natoms, sizeof(_coord_buf[0])));
  }
  _xdr = xdrfile_open(fname.c_str(), mode.c_str());
}

XtcHandler::~XtcHandler() {
  xdrfile_close(_xdr);
  if (_mode == "r") {
    free(_coord_buf);
  }
}

std::vector<float>
XtcHandler::next() {
  if (_mode == "r") {
    int step;
    float time_step;
    float prec;
    matrix box;
    int err = read_xtc(_xdr, _natoms, &step, &time_step, box, _coord_buf, &prec);
    if (err == exdrOK) {
      std::vector<float> v(_natoms*3);
      for (int i=0; i < _natoms; ++i) {
        v[3*i]   = _coord_buf[i][0];
        v[3*i+1] = _coord_buf[i][1];
        v[3*i+2] = _coord_buf[i][2];
      }
      return v;
    }
  }
  _eof = true;
  return {};
}

void
XtcHandler::write(std::vector<float> row) {
  if (_mode == "w") {
    float fake_box_matrix[3][3] = {{0,0,0}, {0,0,0}, {0,0,0}};
    int natoms = row.size() / 3;
    rvec* x = static_cast<rvec*>(calloc(natoms, sizeof(rvec)));
    for (int i=0; i < natoms; ++i) {
      x[i][0] = row[3*i];
      x[i][1] = row[3*i+1];
      x[i][2] = row[3*i+2];
    }
    write_xtc(_xdr, natoms, _nrow, _nrow*1.0f, fake_box_matrix, x, 1000.0f);
    free(x);
    ++_nrow;
  }
}

bool
XtcHandler::eof() {
  return _eof;
}


//// unifying interface

FilePointer
open(std::string fname, std::string mode) {
  if ((fname.size() > 4)
   && (fname.compare(fname.size()-4, 4, ".xtc") == 0)) {
    return FilePointer(new XtcHandler(fname, mode));
  } else {
    return FilePointer(new AsciiHandler(fname, mode));
  }
}

} // end namespace 'CoordsFile'

